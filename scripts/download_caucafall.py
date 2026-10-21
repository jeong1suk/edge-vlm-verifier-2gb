#!/usr/bin/env python3
"""CAUCAFall (CC BY 4.0, Mendeley Data 7w7fccy7ky) 다운로드.

  python scripts/download_caucafall.py labels          # 프레임 라벨(.txt) 전체 (~20k개, 수 KB)
  python scripts/download_caucafall.py frames LIST.txt  # 선택한 프레임 PNG만

파일명 규칙: <활동 접두어><참가자 번호><프레임 5자리>.png  (예: cas300012.png = 참가자 3, 뒤로 낙상, 12번째 프레임)
라벨: 'cls cx cy w h' (YOLO, 정규화 좌표), cls 0=nofall 1=fall
"""
import json, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DATASET = "7w7fccy7ky"
ROOT = Path(__file__).resolve().parent.parent / "data" / "caucafall"
UA = "Mozilla/5.0"  # 기본 python UA는 403

# 엑셀(Dataset details.xlsx)의 활동별 프레임 수와 대조해서 확인한 매핑
ACTIVITY = {
    "cfs": "fall_forward", "cas": "fall_backward", "cis": "fall_left", "cds": "fall_right", "css": "fall_sitting",
    "cams": "walk", "sals": "hop", "res": "pickup", "ses": "sit_down", "ars": "kneel",
}
# 표준 규칙을 따르지 않는 파일: 'cfs1-00001.png'(참가자 1), 'cds5-00001.png'(참가자 5)
NAME_RE = re.compile(r"^(cfs|cas|cis|cds|css|cams|sals|res|ses|ars)(\d{1,2})-?(\d{5})\.(png|txt)$")


def parse(name):
    m = NAME_RE.match(name)
    if not m:
        return None
    p, subj, frame, ext = m.groups()
    return {"activity": ACTIVITY[p], "subject": int(subj), "frame": int(frame), "ext": ext}


def meta():
    cache = ROOT / "meta" / "dataset.json"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["curl", "-sL", "-m", "120", "-A", UA, "-o", str(cache),
                        f"https://data.mendeley.com/public-api/datasets/{DATASET}"], check=True)
    return json.load(open(cache))["files"]


def fetch(args):
    url, dst = args
    if dst.exists() and dst.stat().st_size > 0:
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["curl", "-sL", "-m", "60", "--retry", "3", "-A", UA, "-o", str(dst), url])
    return r.returncode


def main():
    mode = sys.argv[1]
    files = meta()
    if mode == "labels":
        jobs = [(f["content_details"]["download_url"], ROOT / "labels" / f["filename"])
                for f in files if f["filename"].endswith(".txt") and parse(f["filename"])]
    elif mode == "frames":
        want = set(Path(sys.argv[2]).read_text().split())
        jobs = [(f["content_details"]["download_url"], ROOT / "frames" / f["filename"])
                for f in files if f["filename"] in want]
    else:
        raise SystemExit(__doc__)
    print(f"{mode}: {len(jobs)} files", flush=True)
    with ThreadPoolExecutor(32) as ex:
        fails = sum(1 for rc in ex.map(fetch, jobs) if rc)
    print(f"done, failures={fails}", flush=True)


if __name__ == "__main__":
    main()
