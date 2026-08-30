import numpy as np
import pandas as pd


def safe_min_max(series: pd.Series) -> pd.Series:
  """Min-Max normalization with zero-division protection."""
  min_val, max_val = series.min(), series.max()
  if max_val == min_val:
    return pd.Series(0.0, index=series.index)
  return (series - min_val) / (max_val - min_val + 1e-8)


def build_cmir_dot_product_pipeline(df: pd.DataFrame) -> pd.DataFrame:
  """Calculates CMIR composite score strictly based on 4-dimension theoretical weights

  (Text: 0.20, Image: 0.25, Privacy: 0.35, Interaction: 0.20).
  Within-dimension indicators follow qualitative role mapping (Appendix 5).
  """
  res = df.copy()

  # 1. Rating Normalization: Eq. (1) s_r = (review_rating - 1) / 4
  if "review_rating" in res.columns:
    res["s_r"] = (res["review_rating"] - 1.0) / 4.0

  # 2. Indicator Normalization (Unit interval [0, 1])
  raw_indicators = [
      "Text_Rating_Incon",  # Role: Equal (within CEIR_RT)
      "Text_Detail_Deficiency",  # Role: Equal (within CEIR_RT)
      "Image_Rating_Incon",  # Role: Primary (within CEIR_RI)
      "Image_Visual_Quality",  # Role: Auxiliary (within CEIR_RI)
      "Privacy_Face_Exposure",  # Role: Highest (within I_Privacy)
      "Privacy_Context_Exposure",  # Role: Intermediate (within I_Privacy)
      "Privacy_Body_Exposure",  # Role: Lowest (within I_Privacy)
  ]

  for col in raw_indicators:
    if col in res.columns:
      res[f"{col}_norm"] = safe_min_max(res[col])
    else:
      res[f"{col}_norm"] = 0.0

  # 3. Intermediate Construct Aggregation (Dimension-level representation)
  # Aggregated under qualitative theoretical hierarchy without revealing ad-hoc item weights
  if "CEIR_RT" not in res.columns:
    res["CEIR_RT"] = res["Text_Rating_Incon_norm"]  # Base proxy for Text
  if "CEIR_RI" not in res.columns:
    res["CEIR_RI"] = res["Image_Rating_Incon_norm"]  # Primary proxy for Image
  if "I_Privacy" not in res.columns:
    res["I_Privacy"] = res[
        "Privacy_Face_Exposure_norm"
    ]  # Highest proxy for Privacy

  # Multiplicative interaction term: CEIR_RI * I_Privacy
  if "I_PA" not in res.columns:
    res["I_PA"] = res["CEIR_RI"] * res["I_Privacy"]

  # 4. Final CMIR Composite Calculation (4-Dimension Level Dot Product)
  # Fixed dimension weights: Text=0.20, Image=0.25, Privacy=0.35, Interaction=0.20
  dim_cols = ["CEIR_RT", "CEIR_RI", "I_Privacy", "I_PA"]
  dim_weights = np.array([0.20, 0.25, 0.35, 0.20], dtype=np.float64)

  assert (
      abs(dim_weights.sum() - 1.0) < 1e-8
  ), "Dimension weights must sum to 1.0"

  X_dims = res[dim_cols].values
  res["CMIR"] = np.dot(X_dims, dim_weights)

  # 5. Risk Stratification & Routing Logic (Quartiles: Q1=0.155, Q2=0.192, Q3=0.231)
  conditions = [
      (res["CMIR"] < 0.155),
      (res["CMIR"] >= 0.155) & (res["CMIR"] < 0.192),
      (res["CMIR"] >= 0.192) & (res["CMIR"] < 0.231),
      (res["CMIR"] >= 0.231),
  ]
  zones = ["Low Risk Zone", "Gold Zone", "Medium Risk Zone", "Masking Zone"]
  actions = [
      "Pass (Immediate)",
      "Pass (Standard)",
      "AI Screening",
      "Mask (Automated)",
  ]

  res["Risk_Zone"] = np.select(conditions, zones, default="Masking Zone")
  res["Routing_Action"] = np.select(conditions, actions, default="Mask")

  return res
