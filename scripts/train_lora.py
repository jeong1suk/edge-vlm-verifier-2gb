#!/usr/bin/env python3
"""불꽃·연기 LoRA 학습 (D-25).

- 언어 모델만 LoRA (비전·커넥터 그대로 → GGUF 변환 시 mmproj는 원본 사용, D-12와 같은 조건)
- loss는 답 토큰 하나에만: 입력 = 프롬프트(…Assistant:) + " Yes"/" No", 라벨은 마지막 토큰만
  → 평가 하네스의 점수(프롬프트 마지막 위치의 " Yes"/" No" 로짓)와 같은 위치를 학습
- 검증 세트(data/lora/val.jsonl)의 불꽃·연기 AUROC 평균이 가장 좋은 체크포인트를 저장. 평가셋은 보지 않는다.
  python scripts/train_lora.py [--epochs 2] [--lr 1e-4] [--r 16]
"""
import argparse, json, math, os, random, sys, time
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from run_eval import PROMPTS
from metrics import auroc

ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
TARGET = r".*text_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--r", type=int, default=16); ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=1500); ap.add_argument("--out", default="data/lora/adapter")
    ap.add_argument("--data", default="data/lora", help="train.jsonl, val.jsonl, images/ 가 있는 폴더")
    a = ap.parse_args()
    torch.manual_seed(0); random.seed(0)
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from peft import LoraConfig, get_peft_model
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    proc.image_processor.do_image_splitting = False
    proc.image_processor.size = {"longest_edge": 512}
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda")
    model = get_peft_model(model, LoraConfig(r=a.r, lora_alpha=2 * a.r, lora_dropout=0.05, target_modules=TARGET, bias="none"))
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    vis_lora = [n for n, p in model.named_parameters() if p.requires_grad and "vision_model" in n]
    assert not vis_lora, f"비전에 LoRA가 붙음: {vis_lora[:3]}"
    print(f"trainable {n_tr/1e6:.2f}M (text only)", flush=True)
    yes = proc.tokenizer.encode(" Yes", add_special_tokens=False)[0]
    no = proc.tokenizer.encode(" No", add_special_tokens=False)[0]
    D = ROOT / a.data; IMG = D / "images"

    def encode(ex):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPTS[ex["task"]]}]}]
        text = proc.apply_chat_template(msgs, add_generation_prompt=True)
        inp = proc(text=text, images=[Image.open(IMG / ex["image"]).convert("RGB")], return_tensors="pt").to("cuda")
        inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
        return inp

    def val_auroc(rows):
        model.eval(); sc = {"fire": ([], []), "smoke": ([], [])}
        with torch.no_grad():
            for ex in rows:
                lg = model(**encode(ex)).logits[0, -1]
                s = float(torch.softmax(torch.stack([lg[yes], lg[no]]).float(), 0)[0])
                sc[ex["task"]][0 if ex["answer"] == "Yes" else 1].append(s)
        model.train()
        return {t: auroc(p, n) for t, (p, n) in sc.items()}

    train = [json.loads(l) for l in open(D / "train.jsonl")]
    val = [json.loads(l) for l in open(D / "val.jsonl")]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    total = a.epochs * len(train) // a.accum
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 50) * max(0.0, 1 - s / max(1, total)))
    log = {"args": vars(a), "trainable_M": n_tr / 1e6, "n_train": len(train), "n_val": len(val), "evals": []}
    base = val_auroc(val); print("val AUROC before:", base, flush=True); log["evals"].append({"step": 0, **base})
    best, step, t0 = (base["fire"] + base["smoke"]) / 2, 0, time.time()
    model.train()
    for ep in range(a.epochs):
        random.Random(ep).shuffle(train); run = 0.0
        for i, ex in enumerate(train):
            inp = encode(ex)
            ans = yes if ex["answer"] == "Yes" else no
            ids = torch.cat([inp["input_ids"], torch.tensor([[ans]], device="cuda")], 1)
            att = torch.cat([inp["attention_mask"], torch.ones(1, 1, dtype=inp["attention_mask"].dtype, device="cuda")], 1)
            lab = torch.full_like(ids, -100); lab[0, -1] = ans
            out = model(input_ids=ids, attention_mask=att, pixel_values=inp["pixel_values"],
                        pixel_attention_mask=inp.get("pixel_attention_mask"), labels=lab)
            (out.loss / a.accum).backward(); run += out.loss.item()
            if (i + 1) % a.accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); sched.step(); opt.zero_grad(); step += 1
            if (i + 1) % a.eval_every == 0 or i + 1 == len(train):
                v = val_auroc(val); m = (v["fire"] + v["smoke"]) / 2
                rec = {"epoch": ep, "example": i + 1, "step": step, "train_loss": run / a.eval_every, **v, "min": round((time.time() - t0) / 60, 1)}
                log["evals"].append(rec); print(rec, flush=True); run = 0.0
                if m > best:
                    best = m; model.save_pretrained(ROOT / a.out); log["best"] = rec
                    print("  saved best", flush=True)
    (ROOT / a.out).mkdir(parents=True, exist_ok=True)
    if a.data == "data/lora":
        (ROOT / "results" / "lora_train_log.json").write_text(json.dumps(log, indent=1))
    print("done. best:", log.get("best"))


if __name__ == "__main__":
    main()
