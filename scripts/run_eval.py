#!/usr/bin/env python3
"""VLM 2차 검증 평가: 모델·정밀도 하나를 불러와 낙상/불꽃/연기 과제의 이미지별 Yes 확률과 메모리·지연을 기록.

  python scripts/run_eval.py --config fp16            # SmolVLM-500M FP16 (기준)
  python scripts/run_eval.py --config fp16 --full-frame   # 낙상 전체 프레임 비교 (D-05, FP16에서만)
  python scripts/run_eval.py --config bnb8|bnb4|bnb8_all|fp16_256m

레시피는 회사 vlm_worker와 동일 (이전 재현 실험에서 Jetson 기록 점수와 평균 차이 0.0009 확인):
  longest_edge=512, 이미지 분할 없음, 마지막 토큰의 " Yes"/" No" 로짓 2개에 softmax → Yes 확률
메모리 (D-12, Jetson은 CPU·GPU 공유 메모리이므로 합계를 대리 지표로 사용):
  weights    = 모델 로드 직후 torch 할당량
  activation = 추론 중 최대 할당량 - weights
  runtime    = nvidia-smi 프로세스 GPU 사용량 - torch 예약량 (CUDA 컨텍스트, 커널 등)
  rss        = 프로세스 CPU 메모리 (최대). 공유 라이브러리가 포함되므로 USS(이 프로세스만 쓰는 메모리)와
               모델 로드 전 RSS(torch + CUDA 초기화만 한 상태)도 함께 기록
"""
import argparse, json, os, subprocess, sys, threading, time
from pathlib import Path

import psutil
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PROMPTS = {  # 회사 운영 프롬프트(낙상), 이전 실험 프롬프트(불꽃), 같은 형식의 연기 프롬프트 (D-06)
    "fall": "Is this person fallen on the ground? Answer Yes or No.",
    "fire": "Is there visible fire or flame in this scene? Answer Yes or No.",
    "smoke": "Is there visible smoke in this scene? Answer Yes or No.",
}
CONFIGS = {
    "fp16":      ("HuggingFaceTB/SmolVLM-500M-Instruct", None),
    "fp16_256m": ("HuggingFaceTB/SmolVLM-256M-Instruct", None),
    "bnb8":      ("HuggingFaceTB/SmolVLM-500M-Instruct", "int8"),
    "bnb4":      ("HuggingFaceTB/SmolVLM-500M-Instruct", "nf4"),
    "bnb8_all":  ("HuggingFaceTB/SmolVLM-500M-Instruct", "int8_all"),
    "bnb4_all":  ("HuggingFaceTB/SmolVLM-500M-Instruct", "nf4_all"),
}
# D-12: 언어 모델만 양자화. transformers 5는 패턴을 전체 이름의 앞부분부터 맞추므로(re.match)
# "vision_model"이 아니라 "model.vision_model"로 써야 한다 (처음엔 이걸 몰라 비전까지 양자화됨 → check_quant로 방지)
LM_ONLY_SKIP = ["model.vision_model", "model.connector", "lm_head"]
EXPECT_QUANT = {"int8": {"text"}, "nf4": {"text"}, "int8_all": {"text", "vision", "connector"},
                "nf4_all": {"text", "vision", "connector"}}


def gpu_proc_mib():
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    for line in out.strip().splitlines():
        pid, mem = [x.strip() for x in line.split(",")]
        if int(pid) == os.getpid():
            return int(mem)
    return None


class Peak(threading.Thread):
    """추론 중 프로세스 GPU 사용량과 RSS의 최댓값을 0.2초 간격으로 샘플링"""
    def __init__(self):
        super().__init__(daemon=True)
        self.gpu = 0; self.rss = 0; self.uss = 0; self.stop = False
    def run(self):
        p = psutil.Process()
        while not self.stop:
            self.gpu = max(self.gpu, gpu_proc_mib() or 0)
            self.rss = max(self.rss, p.memory_info().rss >> 20)
            time.sleep(0.2)
        self.uss = p.memory_full_info().uss >> 20  # 측정 비용이 커서 종료 시 1회


def load(config):
    from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
    model_id, q = CONFIGS[config]
    proc = AutoProcessor.from_pretrained(model_id)
    proc.image_processor.do_image_splitting = False
    proc.image_processor.size = {"longest_edge": 512}
    kw = {"dtype": torch.float16}
    if q == "int8":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=LM_ONLY_SKIP)
    elif q == "int8_all":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=["lm_head"])
    elif q in ("nf4", "nf4_all"):
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.float16,
                                                       llm_int8_skip_modules=LM_ONLY_SKIP if q == "nf4" else ["lm_head"])
    model = AutoModelForImageTextToText.from_pretrained(model_id, device_map="cuda", **kw).eval()
    return model_id, proc, model, check_quant(model, q)


def check_quant(model, q):
    """부분별 양자화된 Linear 수를 세고, 설정한 조건과 다르면 중단 (조용히 잘못된 조건으로 실행되는 것 방지)"""
    cnt = {}
    for n, m in model.named_modules():
        t = type(m).__name__
        if "Linear" not in t:
            continue
        part = "vision" if ".vision_model." in n else "connector" if ".connector." in n else "text" if ".text_model." in n else n
        key = f"{part}:{'quant' if t != 'Linear' else 'fp16'}"
        cnt[key] = cnt.get(key, 0) + 1
    quantized = {k.split(":")[0] for k in cnt if k.endswith(":quant")}
    expect = EXPECT_QUANT.get(q, set())
    assert quantized == expect, f"양자화 범위 불일치: 기대 {expect}, 실제 {quantized} ({cnt})"
    return cnt


def tasks(full_frame):
    fall = [json.loads(l) for l in open(DATA / "fall" / "manifest.jsonl")]
    fire = [json.loads(l) for l in open(DATA / "fire" / "manifest.jsonl")]
    if full_frame:
        yield "fall_full", "fall", [(r["id"], DATA / "caucafall" / "frames" / r["png"]) for r in fall]
        return
    yield "fall", "fall", [(r["id"], DATA / "fall" / "crops" / f"{r['id']}.png") for r in fall]
    yield "fire", "fire", [(r["id"], DATA / "fire" / "images" / r["image"]) for r in fire]
    yield "smoke", "smoke", [(r["id"], DATA / "fire" / "images" / r["image"]) for r in fire]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=CONFIGS)
    ap.add_argument("--full-frame", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="과제별 앞 N장만 (동작 확인용)")
    ap.add_argument("--images", help="이 폴더의 *.jpg를 낙상 프롬프트로만 채점 (하네스 재현 확인용)")
    a = ap.parse_args()

    torch.cuda.init(); torch.zeros(1, device="cuda")
    ctx_mib = gpu_proc_mib()  # 빈 CUDA 컨텍스트
    rss_before_load = psutil.Process().memory_info().rss >> 20
    model_id, proc, model, quant_modules = load(a.config)
    torch.cuda.synchronize()
    weights_mib = torch.cuda.memory_allocated() / 2**20
    # 가설 확인: GPU로 옮긴 뒤 CPU에 가중치 사본이 남는가? (Jetson은 공유 메모리라 사본이 그대로 낭비가 됨)
    rss_after_load = psutil.Process().memory_info().rss >> 20
    import ctypes, gc
    gc.collect(); ctypes.CDLL("libc.so.6").malloc_trim(0)
    rss_after_trim = psutil.Process().memory_info().rss >> 20
    yes = proc.tokenizer.encode(" Yes", add_special_tokens=False)[0]
    no = proc.tokenizer.encode(" No", add_special_tokens=False)[0]

    def score(img, prompt):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = proc.apply_chat_template(msgs, add_generation_prompt=True)
        inputs = proc(text=text, images=[img], return_tensors="pt").to("cuda")
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1]
        pair = torch.stack([logits[yes], logits[no]]).float()
        return float(torch.softmax(pair, 0)[0]), int(inputs["input_ids"].shape[1])

    for _ in range(3):  # 워밍업 (지연 통계에서 제외)
        score(Image.new("RGB", (640, 480), (128, 128, 128)), PROMPTS["fall"])
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()

    tag = a.config + ("_fullframe" if a.full_frame else "") + (f"_limit{a.limit}" if a.limit else "") + ("_check" if a.images else "")
    out_dir = ROOT / "results" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    peak = Peak(); peak.start()
    summary = {"config": a.config, "model": model_id, "full_frame": a.full_frame,
               "quantized_linear_modules": quant_modules, "tasks": {}}
    with open(out_dir / "scores.jsonl", "w") as fo:
        task_list = ([("check", "fall", [(p.name, p) for p in sorted(Path(a.images).glob("*.jpg"))])]
                     if a.images else tasks(a.full_frame))
        for name, pkey, items in task_list:
            if a.limit:
                items = items[: a.limit]
            lats = []
            for i, (rid, path) in enumerate(items):
                img = Image.open(path).convert("RGB")
                torch.cuda.synchronize(); t0 = time.perf_counter()
                s, ntok = score(img, PROMPTS[pkey])
                torch.cuda.synchronize(); lat = (time.perf_counter() - t0) * 1000
                lats.append(lat)
                fo.write(json.dumps({"task": name, "id": rid, "score": s, "latency_ms": round(lat, 2), "n_tokens": ntok}) + "\n")
                if (i + 1) % 200 == 0:
                    print(f"  {name} {i+1}/{len(items)}", flush=True)
            lats.sort()
            q = lambda p: lats[min(len(lats) - 1, int(p * len(lats)))]
            summary["tasks"][name] = {"n": len(lats), "prompt": PROMPTS[pkey],
                                      "latency_ms": {"p50": round(q(0.5), 1), "p95": round(q(0.95), 1), "mean": round(sum(lats) / len(lats), 1)}}
            print(name, summary["tasks"][name], flush=True)
    peak.stop = True; peak.join()
    reserved = torch.cuda.max_memory_reserved() / 2**20
    act = torch.cuda.max_memory_allocated() / 2**20 - weights_mib
    summary["memory_mib"] = {
        "weights": round(weights_mib), "activation_peak": round(act), "torch_reserved_peak": round(reserved),
        "cuda_context_empty": ctx_mib, "gpu_process_peak": peak.gpu,
        "runtime_overhead": None if not peak.gpu else round(peak.gpu - reserved),
        "cpu_rss_before_model_load": rss_before_load, "cpu_rss_after_load": rss_after_load,
        "cpu_rss_after_gc_trim": rss_after_trim, "cpu_rss_peak": peak.rss, "cpu_uss_end": peak.uss,
        "jetson_proxy_total(gpu+rss)": peak.gpu + peak.rss,
    }
    summary["env"] = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                      "transformers": __import__("transformers").__version__, "batch": 1, "warmup": 3,
                      "note": "RTX 4070 SUPER 데스크톱 수치. Jetson 실측 아님 (D-07)"}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps(summary["memory_mib"], indent=1))


if __name__ == "__main__":
    main()
