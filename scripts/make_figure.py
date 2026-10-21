#!/usr/bin/env python3
"""메모리(대리 합계) vs AUROC (낙상, 연기) → results/fig_memory_vs_auroc.png (라벨은 영어: 폰트 의존성 없이 재현 가능하게)"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
m = json.load(open(ROOT / "results" / "metrics.json"))
pts = {"fp16": "HF FP16", "bnb8": "HF INT8", "bnb4": "HF NF4", "fp16_256m": "HF 256M",
       "gguf_f16": "GGUF F16", "gguf_q8_0": "GGUF Q8_0", "gguf_q4_0": "GGUF Q4_0",
       "gguf_lora_q8_0": "GGUF Q8_0 + LoRA", "gguf_lora_q4_0": "GGUF Q4_0 + LoRA"}
color = lambda k: "tab:green" if "lora" in k else ("tab:blue" if k.startswith("gguf") else ("tab:gray" if "256m" in k else "tab:orange"))
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, task, ci_key, ylim in [(axes[0], "fall", "AUROC_CI95_video", (0.80, 1.0)), (axes[1], "smoke", "AUROC_CI95", (0.78, 0.97))]:
    for k, lab in pts.items():
        mem = m[k]["memory_mib"]["jetson_proxy_total(gpu+rss)"]; a = m[k][task]["AUROC"]; lo, hi = m[k][task][ci_key]
        ax.errorbar(mem, a, yerr=[[a - lo], [hi - a]], fmt="o", color=color(k), capsize=3, alpha=0.85)
        off = (6, -14) if ("lora" in k and task == "fall") else (6, 4)
        ax.annotate(lab, (mem, a), textcoords="offset points", xytext=off, fontsize=8)
    ax.axvline(2048, ls="--", color="red"); ax.text(2060, ylim[0] + 0.01, "2 GiB budget", color="red", fontsize=9)
    ax.set_xlabel("Memory proxy: GPU peak + CPU RSS peak (MiB), RTX 4070 — not Jetson")
    ax.set_ylabel(f"{task.capitalize()} AUROC (95% CI)"); ax.set_ylim(*ylim)
axes[0].set_title("Fall (CAUCAFall, n=500, video-level CI)"); axes[1].set_title("Smoke (D-Fire test, n=750)")
fig.suptitle("SmolVLM-500M 2nd-stage verifier: memory vs accuracy (green = + fire/smoke LoRA)")
fig.tight_layout(); fig.savefig(ROOT / "results" / "fig_memory_vs_auroc.png", dpi=130)
print("saved")
