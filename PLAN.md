# 진행 계획

## 시작 전 체크 (사용자)
- [x] GPU 비우기: ollama `llama-server`(5.6GB), `onvif-rag` uvicorn(3GB) 종료 → `nvidia-smi`로 확인
- [x] D-12(비전 인코더 양자화) 결정
- [ ] git 사용자 이름과 이메일 설정 → 첫 커밋
- [ ] DECISIONS.md의 `내 판단(이유):` 칸 채우기

## 1단계: 데이터 (GPU 불필요)
- [x] D-Fire 테스트 분할 다운로드 (parquet 3개, 615MB)
- [x] CAUCAFall 라벨 전체 (백그라운드 진행 중: `data/dl_cauca_labels.log`)
- [x] CAUCAFall 평가셋 샘플링 → `data/eval_fall.jsonl`
      - 양성: fall 구간을 앞부분(falling)과 뒷부분(collapsed)으로 분할 (D-04)
      - 음성: 줍기(bending), 무릎 꿇기·앉기(squatting), 걷기·뛰기(쉬운 음성)
      - 영상 1개에서 k프레임 (연속 프레임은 서로 비슷하므로 영상 단위 지표도 함께 계산)
      - ❓ 판단 필요: 경계 비율(앞 1/3, 뒤 1/3?), k값
- [x] 선택한 PNG만 다운로드 → 여백 크롭(bbox ×2) 생성 (D-05) + 샘플 육안 확인
- [x] D-Fire 분류: fire / smoke만 / 둘 다 / 없음 → `data/eval_fire.jsonl` (D-06)
      - ❓ 판단 필요: 클래스당 장수 (기본안 각 250장)

## 2단계: FP16 기준선
- [x] 이전 프로젝트의 `run_eval.py` 레시피 이식 (512px, 분할 없음, Yes/No 로짓 softmax)
- [x] SmolVLM-500M FP16: 낙상(크롭) / 낙상(전체 프레임, 1회) / 불꽃 / 연기
- [x] SmolVLM-256M FP16 (D-08)
- [x] 메모리를 가중치 / 실행 환경 / 활성값으로 분리해 측정, 지연 p50/p95 (워밍업 3회, 반복 n 기록)

## 3단계: 양자화
- [x] bitsandbytes INT8 / NF4 (언어 모델만, 비전 포함 여부는 D-12)
- [x] RSS가 늘어나는 원인 확인 → CUDA 지연 로딩 (EAGER 비교, D-22)
- [x] llama.cpp GGUF Q8_0 / Q4_K_M (+Q4_0, 비전 Q8): Yes/No 확률 추출 방법을 먼저 검증 (FP16 GGUF와 transformers FP16 점수가 같은지 확인)
- [x] 이미지별 점수의 FP16 대비 차이, 양성·음성 점수 간격, P/R/F1 sweep, 메모리, 지연

## 4단계: 정리
- [x] threshold 재보정 검증 (사람 단위 분리)
- [x] 결과 표: "2GB 이하 + 정확도 손실 x%p 이내"를 만족하는 조건
- [ ] DECISIONS.md: 뺀 것 목록, 회고, 면접 예상 질문 답
- [ ] README (문제 → 방법 → 검증 → 한계 순서)

## 다음 단계 (이 저장소 범위 밖)
- [ ] Jetson Orin NX 실측 (대리 지표 검증)
- [ ] 회사 도메인 데이터에서 llama.cpp 전처리 차이(D-21) 확인
- [ ] 현장별 threshold 재보정 절차

## 커밋 계획 (저장소를 만든 뒤, 작업 순서대로 나눠서 남김)
git 사용자 정보를 설정한 뒤 아래 순서로 커밋하면 기록이 작업 흐름과 같아진다. (현재는 커밋 0개)
1. `docs: 문제 정의와 판단 기록 시작` — DECISIONS.md(D-01~D-11), PLAN.md, .gitignore
2. `data: CAUCAFall·D-Fire 다운로드 스크립트` — download_caucafall.py, download_dfire.sh
3. `data: 낙상 평가셋 구성(falling/collapsed 분리, 음성은 일상 동작만)` — build_fall_set.py, make_crops.py (D-13~D-15)
4. `data: D-Fire 평가셋 구성, 번호 인접 중복 제거` — build_fire_set.py (D-16, 해시 방식은 뺀 이유와 함께)
5. `eval: HF 하네스와 메모리 3분할 측정` — run_eval.py, requirements-lock.txt (D-18 재현 확인)
6. `fix: bitsandbytes skip 패턴이 비전을 양자화하던 문제 + 양자화 범위 검사` — run_eval.py (D-19)
7. `exp: PyTorch 경로 결과` — run_all.sh, results/{fp16,…,bnb4_all}, metrics.py
8. `eval: llama.cpp GGUF 하네스(캐시 끄기, 512 리사이즈)` — run_gguf.py, run_all_gguf.sh (D-20~D-22)
9. `exp: GGUF 결과와 threshold 재보정` — results/gguf_*, recalibrate.py (D-23)
10. `docs: README, 그림, 결과 정리` — README.md, make_figure.py, summary.md
이후 본인이 고치는 부분은 따로 커밋한다 (예: `docs: AI 사용 범위 직접 작성`).
