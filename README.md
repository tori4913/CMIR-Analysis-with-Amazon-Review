
# 🛡️ CMIR-Multimodal-Evaluator

This repository provides an automated evaluation pipeline for assessing the Cross-Modal Inconsistency Risk (CMIR) in e-commerce multimodal reviews.
The framework facilitates a comparative study by analyzing how CMIR metrics influence the content moderation performance of Large Language Models (LLMs) and supporting statistical modeling for risk-aware screening.

---

## 📁 Repository Structure

```text
cmir-multimodal-evaluator/
├── data/                  # Input directory for review datasets
├── prompts/               # Prompt templates and CMIR risk guidelines
├── Regression_Code.py     # Statistical regression models & econometric analysis
├── ML_Code.py             # Machine learning pipelines & predictive performance tests
├── main.py                # Main execution pipeline for LLM moderation
└── requirements.txt       # Required dependencies

```

---

## 📊 Experimental Setup (Ablation Study)

The pipeline evaluates reviews through two distinct modes to measure the impact of multimodal risk awareness:

* **without_cmir (Control Group 🛑):** Moderation based solely on the specificity and utility of text and image details.
* **with_cmir (Experimental Group 🧪):** Moderation augmented by the CMIR guideline and a relative risk score (0-100), enabling the LLM to assess complex risks such as multimodal inconsistency and privacy exposure.

---

## 📈 Analytical Modules (`Regression_Code.py` & `ML_Code.py`)

To complement the LLM moderation pipeline, the repository includes quantitative evaluation scripts:

1. **`Regression_Code.py`:**
* Conducts econometric and OLS regression models (with Robust standard errors like HC3) to examine the impact of Base Elite variables and multimodal discrepancy scores on review helpfulness (`log_helpful_count`).
* Features automated variable sorting and standardization pipelines for clean comparison between expert (Vine) and general user groups.


2. **`ML_Code.py`:**
* Implements machine learning models to predict and evaluate performance metrics based on text-image inconsistency features, privacy exposures, and user interaction attributes.



---

## 🚀 Getting Started

### 1. Installation 🛠️

Install the necessary dependencies:

```bash
pip install -r requirements.txt

```

### 2. Data Preparation 📂

Place your dataset in the `data/` folder. The system requires an Excel or CSV file with the following standard columns:

* `review_id`: Unique identifier for the review.
* `text`: Content of the review.
* `rating`: Star rating (1-5).
* `cmir`: Precomputed inconsistency risk score.
* `image_url`: Comma-separated image links.

### 3. Execution 🏃‍♂️

* **Run the LLM evaluation pipeline:**
```bash
python main.py

```


* **Run the statistical regression analysis:**
```bash
python Regression_Code.py

```


* **Run the machine learning modeling:**
```bash
python ML_Code.py

```



---

## 📝 License

This project is intended for research purposes. Please ensure compliance with the terms of service for the LLM provider utilized (e.g., Google Gemini API). ⚖️
