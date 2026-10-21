#!/usr/bin/env python3
"""LoRA 후 낙상 점수 이동을 재학습 없이 threshold 재보정만으로 되돌릴 수 있는가 (D-26).

D-24와 같은 방식: 목표 FPR 5%, 참가자 5/5 무작위 분할 200회, 보정 참가자의 음성 영상 25개(125장)로 threshold 결정.
상한선(oracle): 낙상 500장 전체에서 FPR 5%가 되는 threshold → recall (같은 데이터로 고르므로 낙관적인 값)
"""
import json, random
from collections import defaultdict
from metrics import load_scores, ROOT
from calib_size import pick_t, pct

CFGS = [("fp16", "HF FP16"), ("lora_fp16", "HF FP16 + LoRA"), ("gguf_q8_0", "GGUF Q8_0"), ("gguf_lora_q8_0", "GGUF Q8_0 + LoRA")]
R, SEED = 200, 0


def main():
    fall = {r["id"]: r for r in map(json.loads, open(ROOT / "data/fall/manifest.jsonl"))}
    out, L = {}, ["## LoRA 후 낙상: threshold 재보정만으로 되돌릴 수 있는가 (D-26)", "",
                  "목표 FPR 5%. '재보정'은 참가자 5/5 분할 200회(보정 참가자 음성 125장으로 threshold → 나머지 5명에서 측정). '상한선'은 500장 전체에서 고른 threshold.", "",
                  "| config | t=0.55 recall / FPR | 상한선 t | 상한선 recall | 재보정 t 중앙값 | 재보정 recall 중앙값 [10%, 90%] | 재보정 FPR 중앙값 [10%, 90%] |",
                  "|---|---|---|---|---|---|---|"]
    for c, lab in CFGS:
        sc = load_scores(c)["fall"]
        pos = [sc[i] for i, r in fall.items() if r["gt"] == 1]; neg = [sc[i] for i, r in fall.items() if r["gt"] == 0]
        r055 = sum(s >= .55 for s in pos) / len(pos); f055 = sum(s >= .55 for s in neg) / len(neg)
        t_or = pick_t(neg); r_or = sum(s >= t_or for s in pos) / len(pos)
        rng = random.Random(SEED); ts, recs, fprs = [], [], []
        for _ in range(R):
            subs = list(range(1, 11)); rng.shuffle(subs); cal = set(subs[:5])
            t = pick_t([sc[i] for i, r in fall.items() if r["gt"] == 0 and r["subject"] in cal])
            tp = [sc[i] for i, r in fall.items() if r["gt"] == 1 and r["subject"] not in cal]
            tn = [sc[i] for i, r in fall.items() if r["gt"] == 0 and r["subject"] not in cal]
            ts.append(t); recs.append(sum(s >= t for s in tp) / len(tp)); fprs.append(sum(s >= t for s in tn) / len(tn))
        out[c] = {"at_055": [r055, f055], "oracle": {"t": t_or, "recall": r_or},
                  "recal": {"t_median": pct(ts, .5), "recall": [pct(recs, .1), pct(recs, .5), pct(recs, .9)],
                            "fpr": [pct(fprs, .1), pct(fprs, .5), pct(fprs, .9)]}}
        o = out[c]
        L.append(f"| {lab} | {r055:.3f} / {f055:.3f} | {t_or:.3f} | {r_or:.3f} | {o['recal']['t_median']:.3f} | "
                 f"{o['recal']['recall'][1]:.3f} [{o['recal']['recall'][0]:.3f}, {o['recal']['recall'][2]:.3f}] | "
                 f"{o['recal']['fpr'][1]:.3f} [{o['recal']['fpr'][0]:.3f}, {o['recal']['fpr'][2]:.3f}] |")
    (ROOT / "results" / "lora_fall_recal.json").write_text(json.dumps(out, indent=1))
    md = ROOT / "results" / "summary.md"
    s = md.read_text().split("## LoRA 후 낙상")[0].rstrip() + "\n\n" + "\n".join(L) + "\n"
    md.write_text(s); print("\n".join(L))


if __name__ == "__main__":
    main()
