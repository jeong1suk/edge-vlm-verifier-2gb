#!/usr/bin/env python3
"""llama.cpp(GGUF) 경로 평가: llama-server를 설정별로 띄우고 같은 과제(낙상/불꽃/연기)를 채점.

  python scripts/run_gguf.py --config gguf_f16|gguf_q8_0|gguf_q4_k_m|gguf_q4_0|gguf_q8_0_mmq8 [--no-mmap]

점수: /v1/chat/completions, max_tokens=1, top_logprobs=20 → p(" Yes") / (p(" Yes") + p(" No"))
      (전체 vocab softmax에서 두 토큰 비율 = 두 로짓 softmax와 수학적으로 같음 → HF 경로와 같은 정의)
⚠️ 전처리 차이 (D-20): llama.cpp의 Idefics3 전처리는 항상 overview + 타일을 넣는다.
   512px 이하 입력이면 같은 512x512 이미지가 2번 들어간다 (128 이미지 토큰). HF 운영 레시피는 1번(64).
   → GGUF 조건끼리는 gguf_f16을 기준으로 비교하고, HF와의 차이는 실행 환경·전처리 차이로 따로 보고한다.
   입력은 보내기 전에 긴 변을 512px로 줄인다 (HF 레시피의 첫 단계와 같음, Lanczos). 안 줄이면 큰 화재 이미지가
   여러 타일로 쪼개져 이미지 토큰이 158~887개로 달라졌다.
메모리: llama-server 프로세스 하나만 측정 (nvidia-smi 프로세스 GPU 최대 + RSS 최대, USS)
   --cache-ram 0 --no-cache-prompt: 기본값은 끝난 요청의 상태를 RAM에 최대 8GB까지 캐시한다 (첫 실행에서 RSS 9.4GB).
   2차 검증은 매번 새 이미지라 캐시를 다시 쓸 일이 없고, Jetson에서는 그대로 메모리 낭비가 된다.
"""
import argparse, base64, json, math, os, subprocess, sys, threading, time, urllib.request
from pathlib import Path

import io

import psutil
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
from run_eval import PROMPTS, tasks  # 같은 과제·프롬프트 사용

LLAMA = ROOT / "third_party" / "llama"
BIN = LLAMA / "llama-b11770" / "llama-server"
LIBS = f"{LLAMA / 'llama-b11770'}:{LLAMA / 'cudart-llama-b11770-bin-ubuntu-cuda-12.8-x64'}"
G = ROOT / "data" / "gguf"
CONFIGS = {  # (언어 모델, mmproj) — D-12: 기본은 mmproj(비전) F16
    "gguf_f16":       ("SmolVLM-500M-Instruct-f16.gguf", "mmproj-SmolVLM-500M-Instruct-f16.gguf"),
    "gguf_q8_0":      ("SmolVLM-500M-Instruct-Q8_0.gguf", "mmproj-SmolVLM-500M-Instruct-f16.gguf"),
    "gguf_q4_k_m":    ("SmolVLM-500M-Instruct-Q4_K_M.gguf", "mmproj-SmolVLM-500M-Instruct-f16.gguf"),  # 실제 5.89 BPW
    "gguf_q4_0":      ("SmolVLM-500M-Instruct-Q4_0.gguf", "mmproj-SmolVLM-500M-Instruct-f16.gguf"),    # 4.97 BPW
    "gguf_q8_0_mmq8": ("SmolVLM-500M-Instruct-Q8_0.gguf", "mmproj-SmolVLM-500M-Instruct-Q8_0.gguf"),  # 비전까지 Q8
}
PORT = 8765


def gpu_of(pid):
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    for line in out.strip().splitlines():
        p, m = [x.strip() for x in line.split(",")]
        if int(p) == pid:
            return int(m)
    return None


class Peak(threading.Thread):
    def __init__(self, pid):
        super().__init__(daemon=True); self.p = psutil.Process(pid); self.pid = pid
        self.gpu = 0; self.rss = 0; self.stop = False
    def run(self):
        while not self.stop:
            self.gpu = max(self.gpu, gpu_of(self.pid) or 0)
            self.rss = max(self.rss, self.p.memory_info().rss >> 20)
            time.sleep(0.2)


def post(body):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=120))


LONGEST = 512


def encode(path):
    im = Image.open(path).convert("RGB")
    if max(im.size) > LONGEST:
        r = LONGEST / max(im.size)
        im = im.resize((max(1, round(im.width * r)), max(1, round(im.height * r))), Image.LANCZOS)
    buf = io.BytesIO(); im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def score(path, prompt):
    ext = "png"
    b64 = encode(path)
    r = post({"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/{ext};base64,{b64}"}},
        {"type": "text", "text": prompt}]}],
        "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 20, "cache_prompt": False})
    top = {t["token"]: t["logprob"] for t in r["choices"][0]["logprobs"]["content"][0]["top_logprobs"]}
    y, n = top.get(" Yes"), top.get(" No")
    if y is None and n is None:
        s = float("nan")
    elif y is None:
        s = 0.0  # " Yes"가 top-20 밖 → 사실상 0 (기록)
    elif n is None:
        s = 1.0
    else:
        s = 1 / (1 + math.exp(n - y))
    return s, r["usage"]["prompt_tokens"], r.get("timings", {}), (y is None or n is None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=CONFIGS)
    ap.add_argument("--no-mmap", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--images", help="이 폴더의 *.jpg를 낙상 프롬프트로만 채점 (확인용)")
    a = ap.parse_args()
    lm, mm = CONFIGS[a.config]
    cmd = [str(BIN), "-m", str(G / lm), "--mmproj", str(G / mm), "-ngl", "99", "-c", "2048", "-np", "1",
           "--host", "127.0.0.1", "--port", str(PORT), "--no-webui",
           "--cache-ram", "0", "--no-cache-prompt"] + (["--load-mode", "none"] if a.no_mmap else [])  # b11770에서 --no-mmap 대신
    tag = a.config + ("_nommap" if a.no_mmap else "") + (f"_limit{a.limit}" if a.limit else "") + ("_check" if a.images else "")
    out_dir = ROOT / "results" / tag; out_dir.mkdir(parents=True, exist_ok=True)
    log = open(out_dir / "server.log", "w")
    srv = subprocess.Popen(cmd, env={**os.environ, "LD_LIBRARY_PATH": LIBS}, stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(240):
            try:
                if b"ok" in urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2).read():
                    break
            except Exception:
                time.sleep(0.5)
        mem_idle = {"gpu": gpu_of(srv.pid), "rss": psutil.Process(srv.pid).memory_info().rss >> 20}
        warm = ROOT / "data" / "fall" / "crops" / "fall_0000.png"
        for _ in range(3):
            score(warm, PROMPTS["fall"])
        peak = Peak(srv.pid); peak.start()
        summary = {"config": a.config, "lm": lm, "mmproj": mm, "no_mmap": a.no_mmap, "cmd": " ".join(str(x).replace(str(ROOT) + "/", "") for x in cmd[1:]), "tasks": {}}
        task_list = ([("check", "fall", [(p.name, p) for p in sorted(Path(a.images).glob("*.jpg"))])]
                     if a.images else tasks(False))
        with open(out_dir / "scores.jsonl", "w") as fo:
            for name, pkey, items in task_list:
                if a.limit:
                    items = items[: a.limit]
                lats, srv_ms, missing = [], [], 0
                for i, (rid, path) in enumerate(items):
                    t0 = time.perf_counter()
                    s, ntok, tm, miss = score(path, PROMPTS[pkey])
                    lat = (time.perf_counter() - t0) * 1000
                    lats.append(lat); srv_ms.append(tm.get("prompt_ms", 0) + tm.get("predicted_ms", 0)); missing += miss
                    fo.write(json.dumps({"task": name, "id": rid, "score": s, "latency_ms": round(lat, 2),
                                         "server_ms": round(srv_ms[-1], 2), "n_tokens": ntok, "yes_or_no_missing": miss}) + "\n")
                    if (i + 1) % 200 == 0:
                        print(f"  {name} {i+1}/{len(items)}", flush=True)
                lats.sort(); srv_ms.sort()
                q = lambda xs, p: xs[min(len(xs) - 1, int(p * len(xs)))]
                summary["tasks"][name] = {"n": len(lats), "prompt": PROMPTS[pkey], "yes_or_no_missing_in_top20": missing,
                                          "latency_ms": {"p50": round(q(lats, .5), 1), "p95": round(q(lats, .95), 1)},
                                          "server_ms": {"p50": round(q(srv_ms, .5), 1), "p95": round(q(srv_ms, .95), 1)}}
                print(name, summary["tasks"][name], flush=True)
        peak.stop = True; peak.join()
        uss = psutil.Process(srv.pid).memory_full_info().uss >> 20
        summary["memory_mib"] = {"gpu_idle_after_load": mem_idle["gpu"], "rss_idle_after_load": mem_idle["rss"],
                                 "gpu_process_peak": peak.gpu, "cpu_rss_peak": peak.rss, "cpu_uss_end": uss,
                                 "jetson_proxy_total(gpu+rss)": peak.gpu + peak.rss,
                                 "jetson_proxy_total(gpu+uss)": peak.gpu + uss}
        summary["env"] = {"llama.cpp": "b11770 (39da130fc), ubuntu-cuda-12.8-x64 공식 배포판", "gpu": "RTX 4070 SUPER",
                          "note": "latency_ms는 HTTP·base64 포함 클라이언트 측정, server_ms는 서버 내부 처리. Jetson 실측 아님 (D-07)"}
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
        print(json.dumps(summary["memory_mib"], indent=1))
    finally:
        srv.terminate(); srv.wait(timeout=30)


if __name__ == "__main__":
    main()
