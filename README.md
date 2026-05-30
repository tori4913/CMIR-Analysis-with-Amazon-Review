# CMIR-Analysis-with-Amazon-Review

# CMIR Multimodal Review Evaluator

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
