"""
CMIR_Scoring.py
================
Produces the 15 raw sub-indicators (Text_Rating_Incon, Image_Visual_Quality,
Privacy_Face_Exposure, ...) defined in the manuscript's Table 2 and reproduced
verbatim as LLM prompts in Appendix 1.

This is the step that was MISSING from the original repository: the two
prompt files under prompts/ (CMIR_Text_guideline.txt and
CMIR_Image_Privacy_guideline.txt) existed as static text, but no script
actually called the Gemini API with them. Without this script there was no
way to reproduce how the raw indicator columns consumed by
CMIR_Construction.py (Text_Rating_Incon, Image_Visual_Quality,
Privacy_Face_Exposure, etc.) were generated in the first place.

Pipeline position:
    CMIR_Scoring.py  --(raw 15 indicators)-->  CMIR_Construction.py  --(CMIR, Risk_Zone)-->  main.py / Regression_Code.py / ML_Code.py

Input:
    An Excel/CSV file with at least: review_id, product_name, star_rating
    (1-5), review_text, image_url (comma-separated URLs; may be empty).

Output:
    <input>_scored.xlsx containing the original columns plus the raw
    indicator columns needed by CMIR_Construction.build_cmir_dot_product_pipeline().
"""

import argparse
import json
import time
from io import BytesIO

import pandas as pd
import requests
from google import genai
from google.genai import types
from PIL import Image
from tqdm import tqdm

MODEL_NAME = "gemini-2.5-flash-lite"  # matches manuscript Sec. 4.1 / 4.4

TEXT_PROMPT_PATH = "prompts/CMIR_Text_guideline.txt"
IMAGE_PRIVACY_PROMPT_PATH = "prompts/CMIR_Image_Privacy_guideline.txt"

# Fields expected back from each prompt (Appendix 1 [OUTPUT FORMAT] blocks).
TEXT_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "review_id": {"type": "STRING"},
        "Text_Rating_Incon": {"type": "INTEGER"},
        "Text_Rating_Incon_reason": {"type": "STRING"},
        "Text_Logic_Contradiction_count": {"type": "INTEGER"},
        "Text_Logic_Contradiction_reason": {"type": "STRING"},
        "Text_Temporal_Incon": {"type": "INTEGER"},
        "Text_Temporal_Incon_reason": {"type": "STRING"},
        "Text_Certainty_Manip": {"type": "INTEGER"},
        "Text_Certainty_Manip_reason": {"type": "STRING"},
        "Text_Detail_Deficiency": {"type": "INTEGER"},
        "Text_Detail_Deficiency_reason": {"type": "STRING"},
    },
}

IMAGE_PRIVACY_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "review_id": {"type": "STRING"},
        "Image_Visual_Quality": {"type": "INTEGER"},
        "Image_Visual_Quality_reason": {"type": "STRING"},
        "Image_Rating_Incon": {"type": "INTEGER"},
        "Image_Rating_Incon_reason": {"type": "STRING"},
        "Image_Sensory_Diff": {"type": "INTEGER"},
        "Image_Sensory_Diff_reason": {"type": "STRING"},
        "Image_Evidence_Omission": {"type": "INTEGER"},
        "Image_Evidence_Omission_reason": {"type": "STRING"},
        "Image_Text_Conflict": {"type": "INTEGER"},
        "Image_Text_Conflict_reason": {"type": "STRING"},
        "Image_Edit_Intensity": {"type": "INTEGER"},
        "Image_Edit_Intensity_reason": {"type": "STRING"},
        "Privacy_Face_Exposure": {"type": "NUMBER"},
        "Privacy_Face_Exposure_reason": {"type": "STRING"},
        "Privacy_Body_Exposure": {"type": "NUMBER"},
        "Privacy_Body_Exposure_reason": {"type": "STRING"},
        "Privacy_Context_Exposure": {"type": "NUMBER"},
        "Privacy_Context_Exposure_reason": {"type": "STRING"},
        "Privacy_PII_Exposure": {"type": "NUMBER"},
        "Privacy_PII_Exposure_reason": {"type": "STRING"},
    },
}


def load_prompt_template(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def fill_template(template: str, row: pd.Series, image_count: int = 0) -> str:
    return (
        template.replace("{product_name}", str(row.get("product_name", "")))
        .replace("{review_id}", str(row.get("review_id", "")))
        .replace("{star_rating}", str(row.get("star_rating", "")))
        .replace("{review_text}", str(row.get("review_text", "")))
        .replace("{image_count}", str(image_count))
    )


def get_images(row: pd.Series):
    """Download images from a comma-separated 'image_url' column."""
    images = []
    url_data = row.get("image_url")
    if pd.isna(url_data) or not str(url_data).strip():
        return images
    for url in str(url_data).split(","):
        url = url.strip()
        if not url:
            continue
        try:
            res = requests.get(url, timeout=5)
            if res.status_code == 200:
                img = Image.open(BytesIO(res.content)).convert("RGB")
                img.thumbnail((768, 768))
                images.append(img)
        except Exception:
            continue
    return images


def score_text(client, template: str, row: pd.Series, image_count: int) -> dict:
    prompt = fill_template(template, row, image_count)
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json", response_schema=TEXT_SCHEMA
        ),
    )
    return response.parsed or {}


def score_image_privacy(
    client, template: str, row: pd.Series, images: list
) -> dict:
    if not images:
        # No images attached: image/privacy indicators are not applicable.
        return {}
    prompt = fill_template(template, row)
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[prompt] + images,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=IMAGE_PRIVACY_SCHEMA,
        ),
    )
    return response.parsed or {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default="data/reviews_raw.xlsx", help="Input Excel/CSV path"
    )
    parser.add_argument(
        "--output", default="data/reviews_scored.xlsx", help="Output Excel path"
    )
    parser.add_argument(
        "--sleep", type=float, default=1.0, help="Seconds to sleep between reviews"
    )
    args = parser.parse_args()

    api_key = input("Enter Your Gemini API Key: ").strip()
    client = genai.Client(api_key=api_key)

    text_template = load_prompt_template(TEXT_PROMPT_PATH)
    image_privacy_template = load_prompt_template(IMAGE_PRIVACY_PROMPT_PATH)

    if args.input.endswith(".csv"):
        df = pd.read_csv(args.input)
    else:
        df = pd.read_excel(args.input)

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        res_row = row.to_dict()
        images = get_images(row)

        try:
            text_scores = score_text(client, text_template, row, len(images))
        except Exception as e:
            print(f"[WARN] text scoring failed for review_id={row.get('review_id')}: {e}")
            text_scores = {}

        try:
            image_scores = score_image_privacy(
                client, image_privacy_template, row, images
            )
        except Exception as e:
            print(f"[WARN] image/privacy scoring failed for review_id={row.get('review_id')}: {e}")
            image_scores = {}

        for key, val in {**text_scores, **image_scores}.items():
            if key == "review_id":
                continue
            res_row[key] = val

        results.append(res_row)
        pd.DataFrame(results).to_excel(args.output, index=False)  # periodic auto-save
        time.sleep(args.sleep)

    print(f"Done. Wrote {len(results)} scored reviews to {args.output}")


if __name__ == "__main__":
    main()
