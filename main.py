import os
import json
import time
import requests
import pandas as pd
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
MODEL_NAME = "gemini-2.5-flash"

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

def get_prompt(row, mode, img_list):
    """Load prompts based on 'without_cmir' or 'with_cmir' mode."""
    with open(f"prompts/{mode}.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    
    text_content = f"[Review Info]\n- Category: {row.get('subcategory', 'General')}\n- Rating: {row['rating']}\n- Text: {row['text']}"
    
    # Inject CMIR guidelines if mode is 'with_cmir'
    if mode == "with_cmir":
        with open("prompts/cmir_guideline.txt", "r", encoding="utf-8") as f:
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
                res_row[f"{mode}_decision"] = "Error"
        
        results.append(res_row)
        # Periodic auto-save
        pd.DataFrame(results).to_excel("final_results.xlsx", index=False)
        time.sleep(1)

if __name__ == "__main__":
    main()
