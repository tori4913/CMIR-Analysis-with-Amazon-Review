# CMIR-Analysis-with-Amazon-Review

🛡️ CMIR-Multimodal-Evaluator
This repository provides an automated evaluation pipeline for assessing the Cross-Modal Inconsistency Risk (CMIR) in e-commerce multimodal reviews.

The framework facilitates a comparative study by analyzing how CMIR metrics influence the content moderation performance of Large Language Models (LLMs). By providing a structured environment for ablation studies, this tool helps evaluate the necessity of risk-aware screening in downstream AI applications.

## 📁 Repository Structure
```
cmir-multimodal-evaluator/
├── .gitignore
├── README.md
├── requirements.txt
├── main.py
├── .env.example              
├── prompts/                  
│   ├── cmir_guideline.txt
│   ├── version_A.json        # Control group: Prompt without CMIR integration
│   └── version_C.json        # Experimental group: Prompt with CMIR integration
└── data/                     
    └── sample_input.xlsx


📊 Experimental Setup (Ablation Study)
The pipeline evaluates reviews through two distinct modes to measure the impact of multimodal risk awareness:

without_cmir (Control Group 🛑): Moderation based solely on the specificity and utility of text and image details.

with_cmir (Experimental Group 🧪): Moderation augmented by the CMIR guideline and a relative risk score (0-100), enabling the LLM to assess complex risks such as multimodal inconsistency and privacy exposure.

🚀 Getting Started
1. Installation 🛠️
Install the necessary dependencies:

Bash
pip install -r requirements.txt
2. Configuration 🔑
Copy the environment template and set your Gemini API key:

Bash
cp .env.example .env
# Edit the .env file and input your GEMINI_API_KEY
3. Data Preparation 📂
Place your dataset in the data/ folder. The system requires an Excel file with the following standard columns:

review_id: Unique identifier for the review.

text: Content of the review.

rating: Star rating (1-5).

cmir: Precomputed inconsistency risk score.

image_url: Comma-separated image links.

4. Execution 🏃‍♂️
Run the evaluation pipeline:

Bash
python main.py
📝 License
This project is intended for research purposes. Please ensure compliance with the terms of service for the LLM provider utilized (e.g., Google Gemini API). ⚖️
