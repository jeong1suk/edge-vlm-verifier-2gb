#!/usr/bin/env python3
"""D-Fire 테스트 분할에서 평가셋 구성 (D-06, D-16).

분류: fire_only / smoke_only / both / none  (YOLO 클래스: 0=smoke, 1=fire)
각 최대 250장 (fire_only는 테스트 분할에 220장뿐이라 그보다 적음).
D-Fire에는 같은 영상의 연속 프레임(예: AoF08159~08164)이 있어서, 같은 접두어 안에서
파일 번호가 ±GAP 이내인 이미지는 하나만 뽑는다.
- GAP=10은 fire_only를 46장까지 줄였고, WEB(웹 수집) 이미지는 번호가 이어져도 서로 다른 경우가 있었다.
- 지각 해시(dHash 8x8, 평활화+16x16)는 야간 고정 카메라에서 다른 장면을 중복으로, 같은 장면을
  비중복으로 판정해 기준으로 쓸 수 없었다 (data/fire/review_dedup_pairs.jpg).
→ 바로 붙은 번호(±2)만 제외하는 약한 규칙을 쓰고, 남는 상관관계는 출처별 결과로 드러낸다.
이미지는 원본 바이트 그대로 저장 (재인코딩 없음). 박스 면적은 크기별 분석을 위해 기록.
"""
import io, json, random, re
from pathlib import Path
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent / "data"
PER_CLASS, GAP, SEED = 250, 2, 0


def classify(label):
    cls, areas = set(), {"smoke": 0.0, "fire": 0.0}
    for line in (label or "").strip().splitlines():
        p = line.split()
        if len(p) != 5:
            continue
        k = {"0": "smoke", "1": "fire"}[p[0]]
        cls.add(k)
        areas[k] = max(areas[k], float(p[3]) * float(p[4]))
    cat = {frozenset(): "none", frozenset({"fire"}): "fire_only",
           frozenset({"smoke"}): "smoke_only", frozenset({"fire", "smoke"}): "both"}[frozenset(cls)]
    return cat, areas


def key(fn):
    m = re.match(r"^([A-Za-z_]+?)(\d+)", fn)
    return (m.group(1), int(m.group(2))) if m else (fn, -1)


def main():
    meta = []
    for f in sorted((ROOT / "dfire").glob("test-*.parquet")):
        t = pq.read_table(f, columns=["label", "filename"]).to_pylist()
        for i, r in enumerate(t):
            cat, areas = classify(r["label"])
            meta.append({"file": f.name, "row": i, "filename": r["filename"], "cat": cat, "areas": areas})
    counts = {}
    for m in meta:
        counts[m["cat"]] = counts.get(m["cat"], 0) + 1
    print("test split:", len(meta), counts)

    rng = random.Random(SEED)
    chosen = []
    for cat in ["fire_only", "smoke_only", "both", "none"]:
        cand = [m for m in meta if m["cat"] == cat]
        rng.shuffle(cand)
        taken = {}
        for m in cand:
            pre, num = key(m["filename"])
            if num >= 0 and any(abs(num - n) <= GAP for n in taken.get(pre, [])):
                continue
            taken.setdefault(pre, []).append(num)
            chosen.append(m)
            if sum(len(v) for v in taken.values()) == PER_CLASS:
                break
        print(f"{cat}: picked {sum(len(v) for v in taken.values())} / {len(cand)}")

    out = ROOT / "fire" / "images"
    out.mkdir(parents=True, exist_ok=True)
    need = {}
    for m in chosen:
        need.setdefault(m["file"], {})[m["row"]] = m
    for f, rows in need.items():
        t = pq.read_table(ROOT / "dfire" / f, columns=["image"]).column("image")
        for i, m in rows.items():
            (out / m["filename"]).write_bytes(t[i].as_py()["bytes"])
    rows = []
    for i, m in enumerate(chosen):
        rows.append({"id": f"fire_{i:04d}", "image": m["filename"], "cat": m["cat"], "source": key(m["filename"])[0],
                     "gt_fire": int(m["cat"] in ("fire_only", "both")),
                     "gt_smoke": int(m["cat"] in ("smoke_only", "both")),
                     "fire_area": round(m["areas"]["fire"], 5), "smoke_area": round(m["areas"]["smoke"], 5)})
    (ROOT / "fire" / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print("saved", len(rows))


if __name__ == "__main__":
    main()
