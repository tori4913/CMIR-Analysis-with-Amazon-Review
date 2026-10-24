
# CMIR Analysis with Amazon Reviews

This repository provides an automated evaluation pipeline for assessing the Cross-Modal Inconsistency Risk (CMIR) in e-commerce multimodal reviews.
It contains **code, prompts, and reproducibility instructions only**. Raw review text, images, review identifiers, API responses, human-annotation files, and row-level results are deliberately excluded and must remain in an access-controlled local location.

---

## 📁 Repository Structure

```text
CMIR-Analysis-with-Amazon-Review/
├── data/                              # Local input directory only (gitignored; not included)
├── analysis/                          # Revision/reproducibility scripts; code only
│   ├── run_table5_grouped_ml.py        # Table 5 Vine/Non-Vine GroupKFold ML rerun
│   ├── run_ri_exclusion_ols.py         # Reviewer 7-1 RI-only exclusion OLS
│   └── run_table8b_component_attribution.py  # Table 8b component-attribution audit
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
├── requirements.txt                   # Required dependencies
└── .gitignore                         # Blocks data, results, credentials, and caches
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

5. **Revision analyses in `analysis/`** provide code-only reruns requested during revision. `run_table5_grouped_ml.py` performs ASIN-grouped ML evaluation separately for Vine and Non-Vine reviews. `run_ri_exclusion_ols.py` removes Rating–Image Inconsistency from Models 2 and 3 while preserving the remaining group-specific HC3 OLS structure. `run_table8b_component_attribution.py` makes the score-composition and criterion-covariance calculations auditable from the four stored, weighted components.

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

Place an authorized, de-identified local dataset in the `data/` folder. This directory is gitignored and must never be committed. The scripts read local files only and do not upload data.

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

# Reviewer 9-1 / Table 5: ASIN-grouped predictive evaluation
python analysis/run_table5_grouped_ml.py --input data/analysis_input.csv --output-dir outputs/table5 --folds 5 --seed 42

# Reviewer 7-1: RI-only exclusion, Vine/Non-Vine/pooled HC3 OLS
python analysis/run_ri_exclusion_ols.py --input data/analysis_input.csv --output outputs/ri_exclusion_ols.xlsx

# Table 8b: component contribution audit (optional criterion column shown)
python analysis/run_table8b_component_attribution.py --input data/analysis_input.csv --output outputs/table8b_attribution.xlsx --criterion overall_risk_severity_1to5
```

**Model used:** all LLM scoring/moderation calls use `gemini-2.5-flash-lite`, matching the manuscript (Sec. 4.1, 4.4).

## Revision-analysis input requirements

The two revision scripts validate their expected input schema before running. Their exact field lists are declared at the top of each script; this makes data preparation auditable without publishing any data.

- `analysis/run_table5_grouped_ml.py` requires one review-level analysis row with `asin_id`, `amazon_vine`, `log_helpful_count`, the control variables, two text indicators, two image indicators, three privacy indicators, and the specified interaction term.
- `analysis/run_ri_exclusion_ols.py` requires `helpful_vote_log`, `amazon_vine`, the control variables, text variables, `image_visual_quality`, the privacy variables, and `privacy_visual_amplified`. Input column names are normalized to lowercase on import.
- `analysis/run_table8b_component_attribution.py` uses the four **already weighted** component columns (`d_RT`, `d_RI`, `I_privacy`, and `d_Interaction` by default) and, optionally, a locally held validation criterion. It reports covariance-based descriptive attribution; it does not make causal claims.

The Table 5 script saves fold-level and summary CSV files locally. The RI-exclusion script saves a workbook with Vine, Non-Vine, pooled, model-fit, and all-coefficient sheets. Generated outputs are ignored by Git.

## Data protection

Do not commit source review records, raw text, images, URLs, review identifiers, API keys, `.env` files, human-annotation spreadsheets, or generated result files. Before committing, verify with `git status` that only source code, prompts, documentation, and dependency files are staged.

---

## 📝 License

This project is intended for research purposes. Please ensure compliance with the terms of service for the LLM provider utilized (e.g., Google Gemini API). ⚖️
