#!/usr/bin/env python3
"""LoRA 학습 데이터 구성 (D-25). D-Fire 학습 분할만 사용, 평가셋(테스트 분할 854장)은 건드리지 않는다.

유출 방지:
  1) 같은 접두어에서 파일 번호가 평가 이미지와 ±NEIGHBOR 이내인 학습 이미지 제외 (연속 프레임이 분할 경계에 걸칠 수 있음)
  2) 평가 이미지와 바이트가 같은 학습 이미지 제외 (SHA256)
검증 세트: (접두어, 번호//BLOCK) 블록 단위로 약 10%를 떼어 둠 → 블록 안의 연속 프레임이 학습·검증에 나뉘지 않게
샘플: 범주별 최대 CAP장 (fire_only는 적어서 전부). 이미지 1장 → 불꽃 프롬프트 1개 + 연기 프롬프트 1개
출력: data/lora/{train,val}.jsonl, data/lora/images/, data/lora/meta.json
"""
import hashlib, json, random, sys
from collections import Counter
from pathlib import Path
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from build_fire_set import classify, key

ROOT = Path(__file__).resolve().parent.parent / "data"
NEIGHBOR, BLOCK, VAL_FRAC, CAP, SEED = 20, 100, 0.10, 1200, 0


def main():
    test = [json.loads(l) for l in open(ROOT / "fire" / "manifest.jsonl")]
    test_keys = {}
    for r in test:
        p, n = key(r["image"]); test_keys.setdefault(p, []).append(n)
    test_sha = {hashlib.sha256((ROOT / "fire" / "images" / r["image"]).read_bytes()).hexdigest() for r in test}

    rows, excl = [], Counter()
    for f in sorted((ROOT / "dfire").glob("train-*.parquet")):
        t = pq.read_table(f)
        for img, lab, fn in zip(t.column("image").to_pylist(), t.column("label").to_pylist(), t.column("filename").to_pylist()):
            p, n = key(fn)
            if n >= 0 and any(abs(n - m) <= NEIGHBOR for m in test_keys.get(p, [])):
                excl["neighbor"] += 1; continue
            b = img["bytes"]
            if hashlib.sha256(b).hexdigest() in test_sha:
                excl["sha256"] += 1; continue
            cat, _ = classify(lab)
            rows.append({"filename": fn, "cat": cat, "bytes": b, "block": f"{p}:{n // BLOCK}"})
    print("train split usable:", len(rows), dict(Counter(r["cat"] for r in rows)), "excluded:", dict(excl))

    rng = random.Random(SEED)
    blocks = sorted({r["block"] for r in rows}); rng.shuffle(blocks)
    val_blocks = set(blocks[: int(len(blocks) * VAL_FRAC)])
    out = ROOT / "lora"; (out / "images").mkdir(parents=True, exist_ok=True)
    meta = {"neighbor": NEIGHBOR, "block": BLOCK, "val_frac": VAL_FRAC, "cap": CAP, "excluded": dict(excl), "counts": {}}
    for split in ["train", "val"]:
        part = [r for r in rows if (r["block"] in val_blocks) == (split == "val")]
        cap = CAP if split == "train" else int(CAP * VAL_FRAC * 2)
        chosen = []
        for cat in ["fire_only", "smoke_only", "both", "none"]:
            c = [r for r in part if r["cat"] == cat]; rng.shuffle(c); chosen += c[:cap]
        ex = []
        for r in chosen:
            (out / "images" / r["filename"]).write_bytes(r["bytes"])
            gf = int(r["cat"] in ("fire_only", "both")); gs = int(r["cat"] in ("smoke_only", "both"))
            ex.append({"image": r["filename"], "cat": r["cat"], "task": "fire", "answer": "Yes" if gf else "No"})
            ex.append({"image": r["filename"], "cat": r["cat"], "task": "smoke", "answer": "Yes" if gs else "No"})
        rng.shuffle(ex)
        (out / f"{split}.jsonl").write_text("\n".join(json.dumps(e) for e in ex) + "\n")
        meta["counts"][split] = {"images": dict(Counter(r["cat"] for r in chosen)), "examples": len(ex)}
        print(split, meta["counts"][split])
    (out / "meta.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
