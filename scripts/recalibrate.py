#!/usr/bin/env python3
"""threshold 재보정 검증 (D-23): 양자화하면 점수 분포가 움직이므로 t=0.55를 그대로 쓰면 판정이 달라진다.
테스트셋에서 threshold를 고르면 결과가 부풀려지므로 보정용과 평가용을 나눈다.
  낙상: 참가자 1~5 = 보정용, 6~10 = 평가용 (사람 단위 분리)
  화재: 이미지 id 짝수 = 보정용, 홀수 = 평가용
보정 규칙: 보정용 데이터에서 "기준 모델(HF fp16)의 t=0.55 오탐률 이하"를 유지하는 threshold 구간의 **중간값**.
  → 운영 관점: 오탐 예산을 기존과 같게 유지하고, 그 안에서 recall을 비교한다.
  처음에는 구간의 가장 낮은 값을 썼으나, 보정용 음성 점수 바로 위 경계에 붙어 항상 공격적으로 치우쳤다
  (fp16 자기 자신을 재보정해도 t=0.411, 평가용 FPR 0.120→0.200) → 중간값으로 변경.
"""
import json
from pathlib import Path
from metrics import load_scores, ROOT

CFGS = ["fp16", "bnb8", "bnb4", "gguf_f16", "gguf_q8_0", "gguf_q8_0_mmq8", "gguf_q4_k_m", "gguf_q4_0"]


def split_rows():
    fall = {r["id"]: r for r in map(json.loads, open(ROOT / "data/fall/manifest.jsonl"))}
    fire = {r["id"]: r for r in map(json.loads, open(ROOT / "data/fire/manifest.jsonl"))}
    sp = {"fall": {}, "fire": {}, "smoke": {}}
    for i, r in fall.items():
        sp["fall"][i] = ("cal" if r["subject"] <= 5 else "test", r["gt"])
    for i, r in fire.items():
        half = "cal" if int(i.split("_")[1]) % 2 == 0 else "test"
        if r["cat"] in ("fire_only", "both", "none"):
            sp["fire"][i] = (half, r["gt_fire"])
        if r["cat"] in ("smoke_only", "both", "none"):
            sp["smoke"][i] = (half, r["gt_smoke"])
    return sp


def rates(sc, ids, t):
    pos = [sc[i] for i, y in ids if y == 1]; neg = [sc[i] for i, y in ids if y == 0]
    return sum(s >= t for s in pos) / len(pos), sum(s >= t for s in neg) / len(neg)


def main():
    sp = split_rows()
    S = {c: load_scores(c) for c in CFGS}
    out, L = {}, ["## threshold 재보정 (보정용으로 threshold 선택 → 평가용에서 측정)", "",
                  "규칙: 보정용에서 HF fp16의 t=0.55 오탐률 이하를 유지하는 threshold 구간의 중간값. 평가용은 낙상 참가자 6~10, 화재 이미지 절반.", "",
                  "| 과제 | config | t=0.55 R / FPR (평가용) | 재보정 t | 재보정 R / FPR (평가용) |", "|---|---|---|---|---|"]
    for task in ["fall", "fire", "smoke"]:
        cal = [(i, y) for i, (h, y) in sp[task].items() if h == "cal"]
        test = [(i, y) for i, (h, y) in sp[task].items() if h == "test"]
        budget = rates(S["fp16"][task], cal, 0.55)[1]
        for c in CFGS:
            sc = S[c][task]
            negs = sorted(sc[i] for i, y in cal if y == 0)
            k = int(budget * len(negs) + 1e-9)          # 허용 오탐 수
            lo = negs[len(negs) - k - 1]                 # 이 값보다 커야 오탐이 k개 이하
            hi = negs[len(negs) - k] if k > 0 else 1.0   # 이 값 이하이면 오탐 k개 유지
            t = (lo + hi) / 2
            r0, f0 = rates(sc, test, 0.55); r1, f1 = rates(sc, test, t)
            out.setdefault(task, {})[c] = {"t": t, "fpr_budget_cal": budget, "at_055": [r0, f0], "recal": [r1, f1]}
            L.append(f"| {task} | {c} | {r0:.3f} / {f0:.3f} | {t:.3f} | {r1:.3f} / {f1:.3f} |")
    (ROOT / "results" / "recalibration.json").write_text(json.dumps(out, indent=1))
    md = ROOT / "results" / "summary.md"
    s = md.read_text().split("## threshold 재보정")[0].rstrip() + "\n\n" + "\n".join(L) + "\n"
    md.write_text(s); print("\n".join(L))


if __name__ == "__main__":
    main()
