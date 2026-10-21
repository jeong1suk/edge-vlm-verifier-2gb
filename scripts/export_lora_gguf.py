#!/usr/bin/env python3
"""LoRA를 합친 언어 모델을 GGUF로 변환 (D-25).
1) 기본 모델(FP16) + data/lora/adapter → merge_and_unload → data/lora/merged (토크나이저·프로세서 파일 복사)
2) llama.cpp(39da130fc) convert_hf_to_gguf.py --outtype f16 → data/gguf/lora-SmolVLM-500M-f16.gguf
   (같은 변환기로 기본 모델을 변환하면 공식 GGUF와 텐서 291개가 모두 같음을 확인함)
3) llama-quantize → Q8_0, Q4_0. 비전(mmproj)은 LoRA를 안 붙였으므로 원본을 그대로 쓴다.
"""
import os, shutil, subprocess, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
LLAMA = ROOT / "third_party" / "llama"
LIBS = f"{LLAMA / 'llama-b11770'}:{LLAMA / 'cudart-llama-b11770-bin-ubuntu-cuda-12.8-x64'}"
G = ROOT / "data" / "gguf"


def main():
    from transformers import AutoModelForImageTextToText
    from peft import PeftModel
    from huggingface_hub import snapshot_download
    snap = Path(snapshot_download(MODEL_ID))
    merged = ROOT / "data" / "lora" / "merged"
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.float16)
    model = PeftModel.from_pretrained(model, str(ROOT / "data" / "lora" / "adapter")).merge_and_unload()
    model.save_pretrained(merged)
    for f in snap.iterdir():
        if f.suffix in (".json", ".txt") and not (merged / f.name).exists():
            shutil.copy(f, merged / f.name)
    f16 = G / "lora-SmolVLM-500M-f16.gguf"
    subprocess.run([sys.executable, "convert_hf_to_gguf.py", str(merged), "--outtype", "f16", "--outfile", str(f16)],
                   cwd=ROOT / "third_party" / "llama-src", check=True)
    for q in ["Q8_0", "Q4_0"]:
        subprocess.run([str(LLAMA / "llama-b11770" / "llama-quantize"), str(f16), str(G / f"lora-SmolVLM-500M-{q}.gguf"), q],
                       env={**os.environ, "LD_LIBRARY_PATH": LIBS}, check=True, capture_output=True)
    print(sorted(p.name for p in G.glob("lora-*")))


if __name__ == "__main__":
    main()
