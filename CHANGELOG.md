# Changelog — GitHub 체크리스트 반영 내역

원본 저장소(`tori4913/CMIR-Analysis-with-Amazon-Review`)를 논문(Revision2)과
대조해 발견한 문제를 체크리스트 4개 항목 기준으로 수정했습니다.

## [ ] 1. CMIR scoring prompts 업로드

**수정한 파일**
- `prompts/CMIR_Text_guideline.txt` — JSON 출력 예시가 닫는 `}` 없이 잘려 있던 것을 수정.

**새로 만든 파일**
- `CMIR_Scoring.py` — Appendix 1의 두 프롬프트(`CMIR_Text_guideline.txt`,
  `CMIR_Image_Privacy_guideline.txt`)를 실제로 Gemini에 호출해서 15개 raw
  indicator(Text_Rating_Incon, Image_Visual_Quality, Privacy_Face_Exposure 등)를
  생성하는 스크립트. 기존 저장소에는 프롬프트 "텍스트"만 있고 이걸 호출하는
  코드가 전혀 없었음 — §4.1 각주가 말하는 "score-construction procedures"의
  실행 코드가 빠져 있던 부분.
- `prompts/cmir_routing_guideline.txt` — `main.py`의 `with_cmir` 실험군에서
  참조하던 `prompts/cmir_guideline.txt`가 애초에 존재하지 않는 파일이었음.
  라우팅 판단용 짧은 CMIR 해석 가이드로 새로 만들어 연결.

## [ ] 2. CMIR construction/aggregation code 업로드

**수정한 파일**
- `CMIR_Construction.py` — 차원별 가중치가 논문 Table A5-1(Text .20 /
  Image .25 / **Privacy .35** / **Interaction .20**)과 달랐던 부분(코드는
  Privacy .45 / Interaction .10)을 수정. Privacy 하위지표 배분(Body/Face/
  Context)도 논문의 "Highest/Intermediate/Lowest" 순서를 유지한 채 합계
  0.35에 맞게 재조정(0.04 / 0.23 / 0.08). 가중치 합이 1.0인지 확인하는
  `assert` 추가.
  - ⚠️ **선배님 확인 필요:** Risk Zone 경계값(0.155/0.192/0.231)은 그대로
    두었습니다. 이 값은 예전 코드(잘못된 가중치)로 계산한 실제 데이터의
    분위수일 수도 있어서, 가중치를 고친 뒤에도 이 경계값이 여전히 맞는지는
    실제 데이터로 재계산해서 확인이 필요합니다.

## [ ] 3. 공개하기로 한 empirical analysis code 업로드 (robustness 포함)

**새로 만든 파일**
- `Robustness_Code.py` — 논문 본문/부록에는 있지만 코드가 없던 강건성
  분석 4종을 구현:
  1. Segmented regression breakpoint 분석 (Appendix 3, Figure A3)
  2. CMIR 가중치 강건성 — Equal-Weight, RWA(Relative Weights Analysis)
     대안 가중치와 Primary 가중치 간 Pearson r / Kendall tau / Jaccard
     concordance 비교 (Table 12)
  3. Monte Carlo weight perturbation — Dirichlet 샘플링으로 Tight/
     Moderate/Broad 3단계 섭동, N=10,000, seed=42 (Table 13)
  4. 대안 종속변수 모형 — Negative Binomial(NB2), GEE(Poisson, 클러스터
     표준오차, ASIN 그룹) + 시기별(temporal) 강건성 분할 (Appendix 2, §5.1)

  합성 데이터로 4개 섹션 모두 end-to-end 실행 확인함(에러 없이 완주).
  단, GEE는 실제 데이터에 따라 수렴하지 않을 수 있어 실패 시 전체가
  죽지 않고 경고만 출력하도록 방어 코드를 넣었습니다.

## [ ] 4. README/file names와 manuscript repository 문구 일치 확인

**수정한 파일**
- `Rregression_Code.py` → `Regression_Code.py` 로 이름 변경 (README가
  안내하는 `python Regression_Code.py` 명령이 실제로는 파일을 못 찾던 문제).
- `main.py`
  - `MODEL_NAME`을 `"gemini-2.5-flash"` → `"gemini-2.5-flash-lite"`로 수정
    (논문이 §4.1 등 3곳에서 명시한 모델명과 일치시킴).
  - `prompts/{mode}.json` 형태로 소문자 파일명을 열려던 부분을 실제 파일명
    (`with_CMIR.json` / `without_CMIR.json`, 대문자 CMIR)에 맞게 매핑 딕셔너리로
    수정. **이 버그 때문에 README 안내대로 `python main.py`를 실행하면 첫
    호출에서 바로 FileNotFoundError가 났었음** (Linux/GitHub는 대소문자 구분).
  - 예외 발생 시 원인을 콘솔에 출력하도록 로그 추가 (디버깅 편의).
- `README.md` — 전체 재작성:
  - 실제 존재하는 모든 파일(`CMIR_Scoring.py`, `CMIR_Construction.py`,
    `Robustness_Code.py` 포함)을 구조도에 반영.
  - 4단계 파이프라인 실행 순서(Scoring → Construction → 
    Routing/Regression/ML → Robustness)를 명시.
  - 스크립트마다 요구하는 컬럼 스키마가 다른데도 하나로 뭉뚱그려 설명하던
    부분을 스크립트별로 분리해서 명시.
  - 사용 모델명을 `gemini-2.5-flash-lite`로 명시.

---

## 아직 코드만으로는 해결 못 한, 사람이 확인해야 하는 부분

- **CMIR 가중치 수정 후 Risk Zone 경계값 재검증** (위 2번 항목 참고) — 실제
  N=27,912 데이터가 있어야 재계산 가능.
- **`data/` 폴더와 실제 raw 데이터셋은 저장소에 없음** — 이건 처음 안내드린
  대로 확인 대상에서 제외했지만, `CMIR_Scoring.py`를 실행하려면 원본
  리뷰 데이터가 필요합니다.
- 논문 자체의 별개 이슈(§5.1 vs Results의 breakpoint 값 0.235 vs 0.243
  불일치, 한국어 문단 번역, 'Qaulity' 오타 등)는 GitHub 체크리스트와
  무관한 원고(manuscript) 텍스트 문제라 이번 작업 범위에서는 건드리지
  않았습니다.
