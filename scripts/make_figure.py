#!/usr/bin/env python3
"""메모리(대리 합계) vs 낙상 AUROC 그림 → results/fig_memory_vs_auroc.png (라벨은 영어: 폰트 의존성 없이 재현 가능하게)"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
m = json.load(open(ROOT / "results" / "metrics.json"))
pts = {"fp16": "HF FP16", "bnb8": "HF INT8", "bnb4": "HF NF4", "bnb4_all": "HF NF4 (+vision)", "fp16_256m": "HF 256M FP16",
       "gguf_f16": "GGUF F16", "gguf_q8_0": "GGUF Q8_0", "gguf_q8_0_mmq8": "GGUF Q8_0 (+vision Q8)",
       "gguf_q4_k_m": "GGUF Q4_K_M (5.9 BPW)", "gguf_q4_0": "GGUF Q4_0"}
fig, ax = plt.subplots(figsize=(8, 5))
for k, lab in pts.items():
    mem = m[k]["memory_mib"]["jetson_proxy_total(gpu+rss)"]; a = m[k]["fall"]["AUROC"]; lo, hi = m[k]["fall"]["AUROC_CI95_video"]
    c = "tab:blue" if k.startswith("gguf") else ("tab:gray" if "256m" in k else "tab:orange")
    ax.errorbar(mem, a, yerr=[[a - lo], [hi - a]], fmt="o", color=c, capsize=3, alpha=0.85)
    off = {"gguf_q8_0": (8, 10), "gguf_q8_0_mmq8": (-25, 34), "gguf_q4_k_m": (8, -14), "gguf_q4_0": (-25, -26),
           "bnb4_all": (-30, -16), "bnb4": (6, 6)}.get(k, (6, 4))
    ax.annotate(lab, (mem, a), textcoords="offset points", xytext=off, fontsize=8)
ax.axvline(2048, ls="--", color="red"); ax.text(2060, 0.80, "2 GiB budget", color="red", fontsize=9)
ax.set_xlabel("Memory proxy: GPU process peak + CPU RSS peak (MiB) — RTX 4070, not Jetson")
ax.set_ylabel("Fall AUROC (95% CI, video-level bootstrap)")
ax.set_title("SmolVLM-500M 2nd-stage verifier: memory vs accuracy")
ax.set_ylim(0.78, 1.0)
fig.tight_layout(); fig.savefig(ROOT / "results" / "fig_memory_vs_auroc.png", dpi=130)
print("saved")
