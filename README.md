
# 🛡️ CMIR-Multimodal-Evaluator

This repository provides an automated evaluation pipeline for assessing the Cross-Modal Inconsistency Risk (CMIR) in e-commerce multimodal reviews.
The framework facilitates a comparative study by analyzing how CMIR metrics influence the content moderation performance of Large Language Models (LLMs) and supporting statistical modeling for risk-aware screening.

---

## 📁 Repository Structure

```text
CMIR-Analysis-with-Amazon-Review/
├── data/                              # Input directory for review datasets (gitignored)
├── prompts/
│   ├── CMIR_Text_guideline.txt        # Appendix 1: text-axis scoring prompt (15-indicator raw scoring)
│   ├── CMIR_Image_Privacy_guideline.txt  # Appendix 1: image/privacy-axis scoring prompt
│   ├── cmir_routing_guideline.txt     # Compact CMIR interpretation guideline for the 'with_cmir' routing arm
│   ├── with_CMIR.json                 # Sys/task prompt for the 'with_cmir' (experimental) routing arm
│   └── without_CMIR.json              # Sys/task prompt for the 'without_cmir' (control) routing arm
├── CMIR_Scoring.py                    # Step 1: LLM scoring — produces the 15 raw sub-indicators (Appendix 1)
├── CMIR_Construction.py               # Step 2: aggregates raw indicators into the composite CMIR score (Appendix 5)
├── main.py                            # Step 3a: Pass/Mask/Block routing ablation (with_cmir vs without_cmir)
├── Regression_Code.py                 # Step 3b: OLS regression w/ HC3 robust SE (Vine vs Non-Vine)
├── ML_Code.py                         # Step 3c: predictive performance (GBM/RF/LightGBM/XGBoost, GroupKFold by ASIN)
├── Robustness_Code.py                 # Step 4: robustness & sensitivity analyses (see below)
└── requirements.txt                   # Required dependencies
```

---

## 🔗 Pipeline Order

The four analysis scripts are meant to be run in sequence, since each one's output feeds the next:

1. **`CMIR_Scoring.py`** — calls Gemini with the Appendix 1 prompts to produce the 15 raw sub-indicators (`Text_Rating_Incon`, `Image_Visual_Quality`, `Privacy_Face_Exposure`, ...) for every review. Input: a raw review file with `review_id`, `product_name`, `star_rating`, `review_text`, `image_url`. Output: the same rows plus the raw indicator columns.
2. **`CMIR_Construction.py`** — imported as a module (`from CMIR_Construction import build_cmir_dot_product_pipeline`) to aggregate the raw indicators from Step 1 into the composite `CMIR` score, `Risk_Zone`, and `Routing_Action` (Appendix 5).
3. Using the scored + constructed dataset:
   - **`main.py`** runs the Pass/Mask/Block LLM-moderation ablation (`with_cmir` vs `without_cmir`).
   - **`Regression_Code.py`** and **`ML_Code.py`** run the empirical analyses reported in the main results (Table 8–11, Table 5).
4. **`Robustness_Code.py`** runs the sensitivity/robustness analyses reported in the manuscript but not covered by the scripts above: segmented-regression breakpoint (Appendix 3), CMIR weighting robustness incl. RWA (Table 12), Monte Carlo weight perturbation (Table 13), and alternative outcome specifications / temporal split (Appendix 2, Sec. 5.1).

---

## 📊 Experimental Setup (Ablation Study)

The `main.py` pipeline evaluates reviews through two distinct modes to measure the impact of multimodal risk awareness:

* **without_cmir (Control Group 🛑):** Moderation based solely on the specificity and utility of text and image details.
* **with_cmir (Experimental Group 🧪):** Moderation augmented by the CMIR guideline and a relative risk score (0-100), enabling the LLM to assess complex risks such as multimodal inconsistency and privacy exposure.

---

## 📈 Analytical Modules

1. **`Regression_Code.py`** — Conducts econometric OLS regression models (with Robust standard errors, HC3) to examine the impact of Base Elite variables and multimodal discrepancy scores on review helpfulness (`log_helpful_count`). Automated variable sorting/standardization for clean comparison between expert (Vine) and general user groups.
2. **`ML_Code.py`** — Machine learning models predicting review helpfulness from text-image inconsistency features, privacy exposures, and user interaction attributes, evaluated with ASIN-grouped 5-fold cross-validation.
3. **`Robustness_Code.py`** — Segmented-regression breakpoint, RWA/Equal-Weight recalibration of the CMIR weights, Dirichlet Monte Carlo weight perturbation (N = 10,000, seed = 42), Negative Binomial / GEE alternative outcome models, and an early/late temporal split check.

---

## 🚀 Getting Started

### 1. Installation 🛠️

```bash
pip install -r requirements.txt
```

### 2. Data Preparation 📂

Place your dataset in the `data/` folder.

**For `CMIR_Scoring.py`** (raw reviews → raw indicators), the input needs:
* `review_id`, `product_name`, `star_rating` (1–5), `review_text`, `image_url` (comma-separated, may be blank).

**For `main.py`** (LLM moderation ablation), the input needs the scored/constructed output plus:
* `review_id`, `text`, `rating`, `cmir` (composite CMIR score from `CMIR_Construction.py`), `image_url`.

**For `Regression_Code.py` / `ML_Code.py` / `Robustness_Code.py`** (empirical analysis), the input needs one row per review with:
* `asin_id`, `amazon_vine`, `Helpful_Vote_Count` (or precomputed `Y_r = log1p(Helpful_Vote_Count)`), `days_since_review`, `media_count`, `overall_rating`, `review_length`, `s_r`, `verified_purchase`, and the raw `Text_*` / `Image_*` / `Privacy_*` indicator columns produced by `CMIR_Scoring.py`.

### 3. Execution 🏃‍♂️

```bash
# Step 1: score raw text/image/privacy indicators via Gemini
python CMIR_Scoring.py --input data/reviews_raw.xlsx --output data/reviews_scored.xlsx

# Step 2: aggregate into the composite CMIR (used as a library, e.g. in a short driver script)
python -c "import pandas as pd; from CMIR_Construction import build_cmir_dot_product_pipeline as f; \
           f(pd.read_excel('data/reviews_scored.xlsx')).to_excel('data/reviews_with_cmir.xlsx', index=False)"

# Step 3a: LLM moderation ablation (Pass/Mask/Block, with_cmir vs without_cmir)
python main.py

# Step 3b: statistical regression analysis
python Regression_Code.py

# Step 3c: machine learning modeling
python ML_Code.py

# Step 4: robustness & sensitivity analyses
python Robustness_Code.py --input your_data_file.csv
```

**Model used:** all LLM scoring/moderation calls use `gemini-2.5-flash-lite`, matching the manuscript (Sec. 4.1, 4.4).

---

## 📝 License

This project is intended for research purposes. Please ensure compliance with the terms of service for the LLM provider utilized (e.g., Google Gemini API). ⚖️
