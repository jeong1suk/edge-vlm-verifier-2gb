#!/usr/bin/env python3
"""여백을 둔 크롭 생성 (D-05): bbox 중심 기준으로 가로·세로 각각 MARGIN배 확장, 이미지 경계에서 자름.
1차 검출기 크롭을 흉내 내면서 바닥이나 주변 가구 같은 맥락을 남기기 위함.
출력: data/fall/crops/<id>.png (무손실 PNG로 저장해 압축 손실이 변수로 끼지 않게 함)
"""
import json
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent / "data"
MARGIN = 2.0


def crop_box(bbox, W, H, margin=MARGIN):
    cx, cy, w, h = bbox
    x0, x1 = (cx - w * margin / 2) * W, (cx + w * margin / 2) * W
    y0, y1 = (cy - h * margin / 2) * H, (cy + h * margin / 2) * H
    return tuple(int(round(v)) for v in (max(0, x0), max(0, y0), min(W, x1), min(H, y1)))


def main():
    rows = [json.loads(l) for l in open(ROOT / "fall" / "manifest.jsonl")]
    out = ROOT / "fall" / "crops"
    out.mkdir(exist_ok=True)
    sizes = []
    for r in rows:
        im = Image.open(ROOT / "caucafall" / "frames" / r["png"]).convert("RGB")
        box = crop_box(r["bbox"], *im.size)
        c = im.crop(box)
        c.save(out / f"{r['id']}.png")
        sizes.append(c.size)
    ws, hs = sorted(s[0] for s in sizes), sorted(s[1] for s in sizes)
    print(f"{len(sizes)} crops, width min/med/max {ws[0]}/{ws[len(ws)//2]}/{ws[-1]}, "
          f"height {hs[0]}/{hs[len(hs)//2]}/{hs[-1]}")


if __name__ == "__main__":
    main()
