import os
import json
import time
import requests
import pandas as pd
import getpass
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
from google import genai
from google.genai import types
from tqdm import tqdm


# Load API Key
load_dotenv()
API_KEY = getpass.getpass("Enter Your Gemini API Key: ")
client = genai.Client(api_key=API_KEY)
MODEL_NAME = "gemini-2.5-flash-lite"  # manuscript Sec. 4.1 / 4.4: "Gemini 2.5 Flash-Lite"

def get_images(row):
    """Fetch images from 'image_url' column, expecting a list or comma-separated string."""
    images = []
    url_data = row.get("image_url")
    if pd.isna(url_data): return images
    
    # Handle both list and comma-separated string formats
    urls = str(url_data).split(',')
    for url in urls:
        try:
            res = requests.get(url.strip(), timeout=5)
            if res.status_code == 200:
                img = Image.open(BytesIO(res.content)).convert('RGB')
                img.thumbnail((400, 400))
                images.append(img)
        except: continue
    return images

# mode -> actual on-disk filename. NOTE: these filenames use uppercase "CMIR"
# (with_CMIR.json / without_CMIR.json). A previous version of this function
# built the path as f"prompts/{mode}.json" using the lowercase mode string,
# which raised FileNotFoundError on case-sensitive filesystems (Linux/GitHub).
MODE_CONFIG_FILES = {
    "without_cmir": "prompts/without_CMIR.json",
    "with_cmir": "prompts/with_CMIR.json",
}

def get_prompt(row, mode, img_list):
    """Load prompts based on 'without_cmir' or 'with_cmir' mode."""
    with open(MODE_CONFIG_FILES[mode], "r", encoding="utf-8") as f:
        config = json.load(f)
    
    text_content = f"[Review Info]\n- Category: {row.get('subcategory', 'General')}\n- Rating: {row['rating']}\n- Text: {row['text']}"
    
    # Inject the compact CMIR routing guideline if mode is 'with_cmir'.
    # (This is distinct from the Appendix-1 scoring prompts in
    # prompts/CMIR_Text_guideline.txt / CMIR_Image_Privacy_guideline.txt,
    # which are used by CMIR_Scoring.py to produce the raw sub-indicators
    # that feed CMIR_Construction.py in the first place.)
    if mode == "with_cmir":
        with open("prompts/cmir_routing_guideline.txt", "r", encoding="utf-8") as f:
            guide = f.read().replace("{cmir_val}", f"{row['scaled_cmir']:.2f}")
            text_content += f"\n\n{guide}"
            
    prompt = f"{config['sys_prompt']}\n\n[Task]\n{config['task_prompt']}\n\n{text_content}\n\n## Constraints\n{config['output_rule']}"
    return [prompt] + img_list

def main():
    # 1. Load and prepare data
    df = pd.read_excel("data/sample_input.xlsx")
    
    # 2. General Scaling (Min-Max normalization to 0-100)
    df['scaled_cmir'] = ((df['cmir'] - df['cmir'].min()) / (df['cmir'].max() - df['cmir'].min()) * 100).clip(0, 100)
    
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_list = get_images(row)
        res_row = row.to_dict()
        
        # 3. Process both modes
        for mode in ["without_cmir", "with_cmir"]:
            try:
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=get_prompt(row, mode, img_list),
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema={"type": "OBJECT", "properties": {
                            "decision": {"type": "STRING"},
                            "analysis": {"type": "STRING"},
                            "reason": {"type": "STRING"}
                        }}
                    )
                )
                data = response.parsed
                res_row[f"{mode}_decision"] = data.get("decision")
                res_row[f"{mode}_analysis"] = data.get("analysis")
                res_row[f"{mode}_reason"] = data.get("reason")
            except Exception as e:
                print(f"[WARN] mode={mode} review_id={row.get('review_id')} failed: {e}")
                res_row[f"{mode}_decision"] = "Error"
        
        results.append(res_row)
        # Periodic auto-save
        pd.DataFrame(results).to_excel("final_results.xlsx", index=False)
        time.sleep(1)

if __name__ == "__main__":
    main()
