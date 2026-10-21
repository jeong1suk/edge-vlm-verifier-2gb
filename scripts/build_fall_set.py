#!/usr/bin/env python3
"""CAUCAFall 평가셋 구성 (D-04, D-13~D-15).

양성 (fall 영상 50개):
  falling   = fall 라벨 시작 후 +0~+7프레임 중 2장   (20fps 기준 0.35초 이내)
  collapsed = +20프레임(1초) 이후 ~ fall 구간 끝 중 3장 (균등 간격)
  +8~+19는 제외 (전환 구간이라 사람이 봐도 경계가 모호)
음성 (일상 동작 영상 50개, fall 영상의 nofall 프레임은 사용하지 않음):
  bending(pickup), squatting(kneel, sit_down): 자세가 가장 낮은 프레임(bbox 높이 하위 30%)에서 5장
  easy(walk, hop): 영상 전체에서 균등 간격으로 5장
출력: data/fall/manifest.jsonl, data/fall/frame_list.txt
"""
import glob, json, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from download_caucafall import ROOT, parse

OUT = ROOT.parent / "fall"
FALLING = (0, 7)
COLLAPSED_FROM = 20
GROUP = {"pickup": "bending", "kneel": "squatting", "sit_down": "squatting", "walk": "easy", "hop": "easy"}


def even(xs, k):
    """xs에서 양 끝을 포함하지 않고 k개를 균등 간격으로 선택 (결정적, 난수 없음)"""
    if len(xs) <= k:
        return list(xs)
    return [xs[int((i + 0.5) * len(xs) / k)] for i in range(k)]


def main():
    vids = defaultdict(dict)
    for f in glob.glob(str(ROOT / "labels" / "*.txt")):
        name = Path(f).name
        m = parse(name)
        if not m:
            continue
        t = open(f).read().split()
        vids[(m["activity"], m["subject"])][m["frame"]] = {
            "cls": int(t[0]), "bbox": [float(x) for x in t[1:5]], "png": name[:-4] + ".png"}

    rows = []
    for (act, subj), fr in sorted(vids.items()):
        ks = sorted(fr)
        base = {"subject": subj, "activity": act, "video": f"{act}_S{subj}"}
        if act.startswith("fall"):
            fall_ks = [k for k in ks if fr[k]["cls"] == 1]
            on = fall_ks[0]
            # fall 구간이 끊기면 첫 연속 구간만 사용 (fall_forward S9: 뒤 46프레임은 일어나는 장면)
            run = [k for i, k in enumerate(fall_ks) if k - on == i]
            falling = [k for k in run if FALLING[0] <= k - on <= FALLING[1]]
            collapsed = [k for k in run if k - on >= COLLAPSED_FROM]
            for k in even(falling, 2):
                rows.append({**base, **fr[k], "frame": k, "rel": k - on, "gt": 1, "group": "falling"})
            for k in even(collapsed, 3):
                rows.append({**base, **fr[k], "frame": k, "rel": k - on, "gt": 1, "group": "collapsed"})
        else:
            assert all(fr[k]["cls"] == 0 for k in ks), (act, subj)
            g = GROUP[act]
            if g == "easy":
                pick = even(ks, 5)
            else:
                low = sorted(ks, key=lambda k: fr[k]["bbox"][3])[: max(5, int(len(ks) * 0.3))]
                pick = even(sorted(low), 5)
            for k in pick:
                rows.append({**base, **fr[k], "frame": k, "rel": None, "gt": 0, "group": g})

    OUT.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(rows):
        r["id"] = f"fall_{i:04d}"
        del r["cls"]
    (OUT / "manifest.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    (OUT / "frame_list.txt").write_text("\n".join(r["png"] for r in rows) + "\n")
    c = defaultdict(int)
    for r in rows:
        c[(r["gt"], r["group"])] += 1
    print(dict(c), "total", len(rows), "videos", len({r["video"] for r in rows}))


if __name__ == "__main__":
    main()
