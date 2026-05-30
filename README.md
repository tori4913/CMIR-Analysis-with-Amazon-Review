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
│   ├── version_A.json        # (통제군) CMIR 미적용 프롬프트
│   └── version_C.json        # (실험군) CMIR 적용 프롬프트
└── data/                     
    └── sample_input.xlsx
