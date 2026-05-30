공적인 문서의 전문성을 유지하면서도, 가독성을 높여줄 적절한 이모티콘을 활용해 `README.md`를 구성했습니다. 아래 내용을 그대로 복사해서 사용하세요!

---

# 🛡️ CMIR-Multimodal-Evaluator

This repository provides an automated evaluation pipeline for assessing the **Cross-Modal Inconsistency Risk (CMIR)** in e-commerce multimodal reviews.

The framework facilitates a comparative study by analyzing how CMIR metrics influence the content moderation performance of Large Language Models (LLMs). By providing a structured environment for ablation studies, this tool helps evaluate the necessity of risk-aware screening in downstream AI applications.

---

## 📁 Repository Structure

```text
cmir-multimodal-evaluator/
├── data/               # Input directory for review datasets
├── prompts/            # Prompt templates and CMIR risk guidelines
├── main.py             # Main execution pipeline
└── requirements.txt    # Required dependencies

```

---

## 📊 Experimental Setup (Ablation Study)

The pipeline evaluates reviews through two distinct modes to measure the impact of multimodal risk awareness:

* **`without_cmir` (Control Group 🛑)**: Moderation based solely on the specificity and utility of text and image details.
* **`with_cmir` (Experimental Group 🧪)**: Moderation augmented by the CMIR guideline and a relative risk score (0-100), enabling the LLM to assess complex risks such as multimodal inconsistency and privacy exposure.

---

## 🚀 Getting Started

### 1. Installation 🛠️

Install the necessary dependencies:

```bash
pip install -r requirements.txt

```

### 2. Configuration 🔑

Copy the environment template and set your Gemini API key:

```bash
cp .env.example .env
# Edit the .env file and input your GEMINI_API_KEY

```

### 3. Data Preparation 📂

Place your dataset in the `data/` folder. The system requires an Excel file with the following standard columns:

* `review_id`: Unique identifier for the review.
* `text`: Content of the review.
* `rating`: Star rating (1-5).
* `cmir`: Precomputed inconsistency risk score.
* `image_url`: Comma-separated image links.

### 4. Execution 🏃‍♂️

Run the evaluation pipeline:

```bash
python main.py

```

---

## 📝 License

This project is intended for research purposes. Please ensure compliance with the terms of service for the LLM provider utilized (e.g., Google Gemini API). ⚖️
