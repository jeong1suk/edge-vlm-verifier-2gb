#!/usr/bin/env python3
"""results/<config>/scores.jsonl → 지표 표 (results/summary.md, results/metrics.json)

주 지표 AUROC: threshold와 무관하게 비교 가능 (정밀도마다 점수 분포가 달라지므로)
운영 지표 t=0.55: 회사에서 고정해 쓰던 값 → "그대로 바꿔 끼우면 어떻게 되나"
기준 대비: 이미지별 |점수 차| 평균, t=0.55 판정 뒤집힘 비율 (기준은 REF 참고)
신뢰구간: 낙상은 영상 단위(같은 영상 프레임은 서로 비슷함), 화재는 이미지 단위 부트스트랩 1000회, 95%
"""
import json, random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
T = 0.55
ORDER = ["fp16", "fp16_fullframe", "fp16_256m", "bnb8", "bnb4", "bnb8_all", "bnb4_all",
         "gguf_f16", "gguf_q8_0", "gguf_q4_k_m", "gguf_q4_0", "gguf_q8_0_mmq8", "gguf_f16_nommap", "gguf_q4_0_nommap",
         "lora_fp16", "gguf_lora_f16", "gguf_lora_q8_0", "gguf_lora_q4_0"]
# 비교 기준: GGUF 조건은 gguf_f16 (전처리가 HF와 달라서, D-20), 나머지는 HF fp16. gguf_f16 자신은 HF fp16과 비교(실행 환경 차이)
_REF = {"gguf_f16": "fp16", "lora_fp16": "fp16", "gguf_lora_f16": "lora_fp16",
        "gguf_lora_q8_0": "gguf_lora_f16", "gguf_lora_q4_0": "gguf_lora_f16"}
REF = lambda c: _REF.get(c, "gguf_f16" if c.startswith("gguf") else "fp16")


def auroc(pos, neg):
    if not pos or not neg:
        return None
    allv = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg])
    # 동점은 평균 순위
    rank, i, rs = {}, 0, 0.0
    ranks = [0.0] * len(allv)
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        for k in range(i, j + 1):
            ranks[k] = (i + j) / 2 + 1
        i = j + 1
    rp = sum(r for r, (_, y) in zip(ranks, allv) if y == 1)
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def prf(pos, neg, t=T):
    tp = sum(s >= t for s in pos); fn = len(pos) - tp; fp = sum(s >= t for s in neg)
    p = tp / (tp + fp) if tp + fp else 0.0; r = tp / len(pos) if pos else 0.0
    return {"P": p, "R": r, "F1": 2 * p * r / (p + r) if p + r else 0.0, "FPR": fp / len(neg) if neg else 0.0}


def boot_ci(clusters_pos, clusters_neg, n=1000, seed=0):
    """clusters_*: {cluster_id: [scores]} → AUROC 95% 구간 (클러스터 단위 재표집)"""
    rng = random.Random(seed); kp, kn = list(clusters_pos), list(clusters_neg); vals = []
    for _ in range(n):
        pos = [s for k in (rng.choice(kp) for _ in kp) for s in clusters_pos[k]]
        neg = [s for k in (rng.choice(kn) for _ in kn) for s in clusters_neg[k]]
        vals.append(auroc(pos, neg))
    vals.sort()
    return [vals[int(0.025 * n)], vals[int(0.975 * n)]]


def load_scores(cfg):
    p = ROOT / "results" / cfg / "scores.jsonl"
    if not p.exists():
        return None
    d = defaultdict(dict)
    for l in open(p):
        r = json.loads(l); d[r["task"]][r["id"]] = r["score"]
    return d


def main():
    fall = {r["id"]: r for r in map(json.loads, open(ROOT / "data/fall/manifest.jsonl"))}
    fire = {r["id"]: r for r in map(json.loads, open(ROOT / "data/fire/manifest.jsonl"))}
    cfgs = [c for c in ORDER if (ROOT / "results" / c / "scores.jsonl").exists()]
    S = {c: load_scores(c) for c in cfgs}
    out = {}
    for c in cfgs:
        m = {}
        summ = json.load(open(ROOT / "results" / c / "summary.json"))
        m["memory_mib"] = summ.get("memory_mib")
        # GGUF는 서버 내부 처리 시간(server_ms)을 지연으로 사용 (HTTP·base64 제외 → HF 측정과 범위를 맞춤)
        m["latency_ms"] = {k: v.get("server_ms", v["latency_ms"]) for k, v in summ["tasks"].items()}
        fk = "fall_full" if c.endswith("fullframe") else "fall"
        if fk in S[c]:
            sc = S[c][fk]
            pos = [sc[i] for i in sc if fall[i]["gt"] == 1]; neg = [sc[i] for i in sc if fall[i]["gt"] == 0]
            cp, cn = defaultdict(list), defaultdict(list)
            for i, s in sc.items():
                (cp if fall[i]["gt"] else cn)[fall[i]["video"]].append(s)
            g = {}
            for grp in ["falling", "collapsed", "bending", "squatting", "easy"]:
                xs = [sc[i] for i in sc if fall[i]["group"] == grp]
                g[grp] = {"n": len(xs), "mean_score": sum(xs) / len(xs), "rate_ge_t": sum(x >= T for x in xs) / len(xs)}
            m["fall"] = {"AUROC": auroc(pos, neg), "AUROC_CI95_video": boot_ci(cp, cn),
                         "AUROC_falling_vs_neg": auroc([sc[i] for i in sc if fall[i]["group"] == "falling"], neg),
                         "AUROC_collapsed_vs_neg": auroc([sc[i] for i in sc if fall[i]["group"] == "collapsed"], neg),
                         "at_t": prf(pos, neg), "groups": g}
        for task, gtk, other in [("fire", "gt_fire", "smoke_only"), ("smoke", "gt_smoke", "fire_only")]:
            if task not in S[c]:
                continue
            sc = S[c][task]
            pos = [sc[i] for i in sc if fire[i][gtk] == 1]; neg = [sc[i] for i in sc if fire[i]["cat"] == "none"]
            oth = [sc[i] for i in sc if fire[i]["cat"] == other]
            area = "fire_area" if task == "fire" else "smoke_area"
            bins = {"small(<1%)": (0, 0.01), "mid(1-10%)": (0.01, 0.10), "large(>=10%)": (0.10, 9)}
            by = {}
            for b, (lo, hi) in bins.items():
                xs = [sc[i] for i in sc if fire[i][gtk] == 1 and lo <= fire[i][area] < hi]
                by[b] = {"n": len(xs), "AUROC_vs_none": auroc(xs, neg)}
            src = {}
            for so in ["AoF", "WEB", "PublicDataset"]:
                ps = [sc[i] for i in sc if fire[i][gtk] == 1 and fire[i]["source"] == so]
                ns = [sc[i] for i in sc if fire[i]["cat"] == "none" and fire[i]["source"] == so]
                src[so] = {"n_pos": len(ps), "n_neg": len(ns), "AUROC": auroc(ps, ns)}
            m[task] = {"AUROC": auroc(pos, neg), "by_source": src,
                       "AUROC_CI95": boot_ci({i: [s] for i, s in enumerate(pos)}, {i: [s] for i, s in enumerate(neg)}),
                       "at_t": prf(pos, neg), f"{other}_rate_ge_t": sum(x >= T for x in oth) / len(oth),
                       "by_box_area": by}
        ref = S.get(REF(c))
        if ref and c != "fp16":
            agree = {}
            for task in S[c]:
                rt = "fall" if task == "fall_full" else task
                if task == "fall_full":
                    continue  # 입력이 달라서 FP16 크롭과 이미지별 비교는 의미 없음
                ds = [abs(S[c][task][i] - ref[rt][i]) for i in S[c][task]]
                flips = sum((S[c][task][i] >= T) != (ref[rt][i] >= T) for i in S[c][task])
                agree[task] = {"mean_abs_diff": sum(ds) / len(ds), "max_abs_diff": max(ds), "flip_rate": flips / len(ds)}
            m["vs_ref"] = {"ref": REF(c), **agree}
        out[c] = m
    (ROOT / "results" / "metrics.json").write_text(json.dumps(out, indent=1))

    f = lambda x: "-" if x is None else f"{x:.3f}"
    L = ["# 결과 요약 (자동 생성: scripts/metrics.py)", "",
         "RTX 4070 SUPER, batch 1, 워밍업 3회. 메모리 합계는 Jetson 공유 메모리의 **대리 지표**(GPU 프로세스 최대 + CPU RSS 최대)이며 Jetson 실측이 아님.", "",
         "## 정확도 (AUROC [95% CI]) 와 t=0.55 판정", "",
         "| config | 낙상 AUROC | falling/collapsed AUROC | 낙상 R / FPR @0.55 | 불꽃 AUROC | 불꽃 R / FPR @0.55 | 연기만→불꽃 Yes | 연기 AUROC | 연기 R / FPR @0.55 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for c, m in out.items():
        fa, fi, sm = m.get("fall"), m.get("fire"), m.get("smoke")
        L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            c,
            f"{f(fa['AUROC'])} [{f(fa['AUROC_CI95_video'][0])}, {f(fa['AUROC_CI95_video'][1])}]" if fa else "-",
            f"{f(fa['AUROC_falling_vs_neg'])} / {f(fa['AUROC_collapsed_vs_neg'])}" if fa else "-",
            f"{f(fa['at_t']['R'])} / {f(fa['at_t']['FPR'])}" if fa else "-",
            f"{f(fi['AUROC'])} [{f(fi['AUROC_CI95'][0])}, {f(fi['AUROC_CI95'][1])}]" if fi else "-",
            f"{f(fi['at_t']['R'])} / {f(fi['at_t']['FPR'])}" if fi else "-",
            f(fi["smoke_only_rate_ge_t"]) if fi else "-",
            f"{f(sm['AUROC'])} [{f(sm['AUROC_CI95'][0])}, {f(sm['AUROC_CI95'][1])}]" if sm else "-",
            f"{f(sm['at_t']['R'])} / {f(sm['at_t']['FPR'])}" if sm else "-"))
    L += ["", "## 메모리 (MiB) 와 지연 (낙상 크롭, ms)", "",
          "| config | 가중치 | 활성값 최대 | CUDA 컨텍스트 | GPU 프로세스 최대 | RSS 로드 전 | RSS 로드 후 | RSS gc+trim 후 | RSS 최대 | **대리 합계** | p50 | p95 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c, m in out.items():
        if c.startswith("gguf"):
            continue
        mm = m["memory_mib"] or {}; lt = m["latency_ms"].get("fall") or m["latency_ms"].get("fall_full") or {}
        L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | **%s** | %s | %s |" % (
            c, mm.get("weights"), mm.get("activation_peak"), mm.get("cuda_context_empty"), mm.get("gpu_process_peak"),
            mm.get("cpu_rss_before_model_load"), mm.get("cpu_rss_after_load"), mm.get("cpu_rss_after_gc_trim"),
            mm.get("cpu_rss_peak"), mm.get("jetson_proxy_total(gpu+rss)"), lt.get("p50"), lt.get("p95")))
    g = [(c, m) for c, m in out.items() if c.startswith("gguf")]
    if g:
        L += ["", "## 메모리 (MiB) 와 지연 — llama.cpp GGUF (llama-server 프로세스 단독, 서버 내부 처리 시간)", "",
              "| config | 로드 후 GPU | 로드 후 RSS | GPU 최대 | RSS 최대 | USS | **대리 합계(GPU+RSS)** | p50 | p95 |", "|---|---|---|---|---|---|---|---|---|"]
        for c, m in g:
            mm = m["memory_mib"]; lt = m["latency_ms"].get("fall", {})
            L.append(f"| {c} | {mm['gpu_idle_after_load']} | {mm['rss_idle_after_load']} | {mm['gpu_process_peak']} | {mm['cpu_rss_peak']} | {mm['cpu_uss_end']} | **{mm['jetson_proxy_total(gpu+rss)']}** | {lt.get('p50')} | {lt.get('p95')} |")
    L += ["", "## 기준 대비 (같은 이미지, 같은 프롬프트)", "",
          "기준: HF 경로는 fp16, GGUF 경로는 gguf_f16. gguf_f16 행은 HF fp16과의 비교(실행 환경·전처리 차이).", "",
          "| config | 기준 | 과제 | 평균 |점수 차| | 최대 | 판정 뒤집힘 @0.55 |", "|---|---|---|---|---|---|"]
    for c, m in out.items():
        a0 = dict(m.get("vs_ref") or {}); ref = a0.pop("ref", "")
        for task, a in a0.items():
            L.append(f"| {c} | {ref} | {task} | {a['mean_abs_diff']:.4f} | {a['max_abs_diff']:.4f} | {a['flip_rate']*100:.1f}% |")
    L += ["", "## 낙상 그룹별 (평균 점수 / 점수 ≥0.55 비율)", "", "| config | falling | collapsed | bending | squatting | easy |", "|---|---|---|---|---|---|"]
    for c, m in out.items():
        if "fall" in m:
            g = m["fall"]["groups"]
            L.append("| " + c + " | " + " | ".join(f"{g[k]['mean_score']:.2f} / {g[k]['rate_ge_t']*100:.0f}%" for k in ["falling", "collapsed", "bending", "squatting", "easy"]) + " |")
    L += ["", "## 화재: 박스 면적별 AUROC (양성 vs none)", "", "| config | 과제 | small(<1%) | mid(1-10%) | large(>=10%) |", "|---|---|---|---|---|"]
    for c, m in out.items():
        for task in ["fire", "smoke"]:
            if task in m:
                b = m[task]["by_box_area"]
                L.append(f"| {c} | {task} | " + " | ".join(f"{f(v['AUROC_vs_none'])} (n={v['n']})" for v in b.values()) + " |")
    L += ["", "## 화재: 출처별 AUROC (양성 vs none, 같은 출처끼리)", "",
          "AoF = 고정 감시카메라(학습 분할과 같은 카메라가 있을 수 있음), WEB = 웹 수집. PublicDataset은 음성이 4장뿐이라 생략.", "",
          "| config | 과제 | AoF | WEB |", "|---|---|---|---|"]
    for c, m in out.items():
        for task in ["fire", "smoke"]:
            if task in m:
                b = m[task]["by_source"]
                L.append(f"| {c} | {task} | {f(b['AoF']['AUROC'])} (+{b['AoF']['n_pos']}/-{b['AoF']['n_neg']}) | {f(b['WEB']['AUROC'])} (+{b['WEB']['n_pos']}/-{b['WEB']['n_neg']}) |")
    (ROOT / "results" / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
