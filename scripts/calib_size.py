#!/usr/bin/env python3
"""현장 재보정에 필요한 데이터 양 (D-24).

질문: 보정용 데이터를 얼마나 모으면, 목표로 한 오탐률(FPR)이 처음 보는 사람·장면에서도 지켜지는가?
낙상: 참가자 10명을 무작위로 5/5 분할 → 보정 후보 참가자에서 양성·음성 영상을 k개씩 뽑아 threshold 결정
      → 평가용 참가자 전체에서 FPR, recall 측정. 200회 반복. (CAUCAFall은 참가자마다 조명이 다름: 자연광/인공광/0lux)
화재·연기: 이미지를 무작위로 반씩 나누고, 보정용 절반에서 클래스당 k장을 뽑아 같은 방식으로 측정.
threshold 규칙: 보정용 음성 점수에서 허용 오탐 수(floor(target*n))를 넘지 않는 구간의 중간값 (D-23과 같은 규칙).
출력: results/calib_size.json, results/summary.md 마지막 절
"""
import json, random
from collections import defaultdict
from metrics import load_scores, ROOT

TARGET, R, SEED = 0.05, 200, 0
CFGS = ["fp16", "gguf_q8_0", "gguf_q4_0", "bnb4"]
KS = {"fall": [2, 5, 10, 25], "fire": [10, 25, 50, 100], "smoke": [10, 25, 50, 100]}


def pick_t(neg):
    neg = sorted(neg); n = len(neg); k = int(TARGET * n + 1e-9)
    lo = neg[n - k - 1]; hi = neg[n - k] if k > 0 else 1.0
    return (lo + hi) / 2


def pct(xs, p):
    xs = sorted(xs); return xs[min(len(xs) - 1, int(p * len(xs)))]


def main():
    fall = {r["id"]: r for r in map(json.loads, open(ROOT / "data/fall/manifest.jsonl"))}
    fire = {r["id"]: r for r in map(json.loads, open(ROOT / "data/fire/manifest.jsonl"))}
    S = {c: load_scores(c) for c in CFGS}
    out = {}
    # 낙상: 영상 단위
    vids = defaultdict(list)
    for i, r in fall.items():
        vids[(r["subject"], r["video"], r["gt"])].append(i)
    for c in CFGS:
        sc = S[c]["fall"]
        for k in KS["fall"]:
            rng = random.Random(SEED); fprs, recs = [], []
            for _ in range(R):
                subs = list(range(1, 11)); rng.shuffle(subs); cal_s, test_s = set(subs[:5]), set(subs[5:])
                pv = [v for v in vids if v[0] in cal_s and v[2] == 1]; nv = [v for v in vids if v[0] in cal_s and v[2] == 0]
                cal_neg = [sc[i] for v in rng.sample(nv, k) for i in vids[v]]
                t = pick_t(cal_neg)
                tp = [sc[i] for v in vids if v[0] in test_s and v[2] == 1 for i in vids[v]]
                tn = [sc[i] for v in vids if v[0] in test_s and v[2] == 0 for i in vids[v]]
                fprs.append(sum(s >= t for s in tn) / len(tn)); recs.append(sum(s >= t for s in tp) / len(tp))
            out.setdefault("fall", {}).setdefault(c, {})[k] = {"FPR_median": pct(fprs, .5), "FPR_p10": pct(fprs, .1), "FPR_p90": pct(fprs, .9),
                                                              "P_FPR_over_2x_target": sum(f > 2 * TARGET for f in fprs) / R,
                                                              "recall_median": pct(recs, .5), "recall_p10": pct(recs, .1)}
    # 화재·연기: 이미지 단위
    for task, gk, keep in [("fire", "gt_fire", ("fire_only", "both", "none")), ("smoke", "gt_smoke", ("smoke_only", "both", "none"))]:
        ids = [i for i, r in fire.items() if r["cat"] in keep]
        for c in CFGS:
            sc = S[c][task]
            for k in KS[task]:
                rng = random.Random(SEED); fprs, recs = [], []
                for _ in range(R):
                    sh = ids[:]; rng.shuffle(sh); cal, test = sh[: len(sh) // 2], sh[len(sh) // 2:]
                    neg = [i for i in cal if fire[i][gk] == 0]
                    t = pick_t([sc[i] for i in rng.sample(neg, min(k, len(neg)))])
                    tn = [sc[i] for i in test if fire[i][gk] == 0]; tp = [sc[i] for i in test if fire[i][gk] == 1]
                    fprs.append(sum(s >= t for s in tn) / len(tn)); recs.append(sum(s >= t for s in tp) / len(tp))
                out.setdefault(task, {}).setdefault(c, {})[k] = {"FPR_median": pct(fprs, .5), "FPR_p10": pct(fprs, .1), "FPR_p90": pct(fprs, .9),
                                                                "P_FPR_over_2x_target": sum(f > 2 * TARGET for f in fprs) / R,
                                                                "recall_median": pct(recs, .5), "recall_p10": pct(recs, .1)}
    (ROOT / "results" / "calib_size.json").write_text(json.dumps(out, indent=1))
    unit = {"fall": "음성 영상 수(1개 = 5장)", "fire": "음성 이미지 수", "smoke": "음성 이미지 수"}
    L = ["## 현장 재보정에 필요한 데이터 양 (목표 FPR 5%, 200회 반복)", "",
         "낙상: 참가자를 무작위로 5/5 분할해 보정 참가자의 음성 영상 k개로 threshold를 정하고, 처음 보는 참가자 5명에서 측정. 화재·연기: 이미지 절반 분할.",
         "FPR은 중앙값 [10%, 90% 분위]. '목표 2배 초과'는 평가용 FPR이 10%를 넘은 비율.", "",
         "| 과제 | config | k (" + "/".join(unit.values()) + ") | 평가용 FPR | 목표 2배 초과 | recall 중앙값 [10% 분위] |", "|---|---|---|---|---|---|"]
    for task in ["fall", "fire", "smoke"]:
        for c in CFGS:
            for k, v in out[task][c].items():
                L.append(f"| {task} | {c} | {k} | {v['FPR_median']:.3f} [{v['FPR_p10']:.3f}, {v['FPR_p90']:.3f}] | {v['P_FPR_over_2x_target']*100:.0f}% | {v['recall_median']:.3f} [{v['recall_p10']:.3f}] |")
    md = ROOT / "results" / "summary.md"
    s = md.read_text().split("## 현장 재보정에 필요한 데이터 양")[0].rstrip() + "\n\n" + "\n".join(L) + "\n"
    md.write_text(s); print("\n".join(L))


if __name__ == "__main__":
    main()
