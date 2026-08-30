import numpy as np
import pandas as pd


def safe_min_max(series: pd.Series) -> pd.Series:
  """Min-Max normalization with zero-division protection."""
  min_val, max_val = series.min(), series.max()
  if max_val == min_val:
    return pd.Series(0.0, index=series.index)
  return (series - min_val) / (max_val - min_val + 1e-8)


def build_cmir_dot_product_pipeline(df: pd.DataFrame) -> pd.DataFrame:
  """Calculates CMIR by applying a 1:1 dot product with exact paper sub-indicator weight vectors."""
  res = df.copy()

  # 1. Rating Normalization: Eq. (1) s_r = (review_rating - 1) / 4
  if 'review_rating' in res.columns:
    res['s_r'] = (res['review_rating'] - 1.0) / 4.0

  # 2. Paper-Adopted Sub-Indicators Normalization
  # NOTE (fixed 2026-08): weights below now match manuscript Table A5-1 /
  # Sec. 4.5.2 dimension weights (Text .20 / Image .25 / Privacy .35 /
  # Interaction .20). The previous version of this file used Privacy=.45
  # and Interaction=.10, which did not match the manuscript.
  sub_indicators = [
      'Text_Rating_Incon',  # w1 = 0.10  (within CEIR_Text, "Equal")
      'Text_Detail_Deficiency',  # w2 = 0.10  (within CEIR_Text, "Equal")
      'Image_Rating_Incon',  # w3 = 0.15  (within CEIR_Image, "Primary")
      'Image_Visual_Quality',  # w4 = 0.10  (within CEIR_Image, "Auxiliary")
      'Privacy_Body_Exposure',  # w5 = 0.04  (within Privacy, "Lowest")
      'Privacy_Face_Exposure',  # w6 = 0.23  (within Privacy, "Highest")
      'Privacy_Context_Exposure',  # w7 = 0.08  (within Privacy, "Intermediate")
  ]

  for col in sub_indicators:
    if col in res.columns:
      res[f'{col}_norm'] = safe_min_max(res[col])

  # 3. Intermediate Construct Aggregation (4 Main Dimensions)
  res['CEIR_RT'] = res.get('Text_Rating_Incon_norm', 0) * 0.5 + res.get(
      'Text_Detail_Deficiency_norm', 0
  ) * 0.5
  res['CEIR_RI'] = (
      res.get('Image_Rating_Incon_norm', 0) * (0.15 / 0.25)
      + res.get('Image_Visual_Quality_norm', 0) * (0.10 / 0.25)
  )
  res['I_Privacy'] = (
      res.get('Privacy_Body_Exposure_norm', 0) * (0.04 / 0.35)
      + res.get('Privacy_Face_Exposure_norm', 0) * (0.23 / 0.35)
      + res.get('Privacy_Context_Exposure_norm', 0) * (0.08 / 0.35)
  )
  res['I_PA'] = res['CEIR_RI'] * res['I_Privacy']

  # 4. Dot Product Strategy for Composite CMIR Calculation
  # Feature Matrix (X): Shape (N, 8)
  feature_cols = [
      'Text_Rating_Incon_norm',
      'Text_Detail_Deficiency_norm',
      'Image_Rating_Incon_norm',
      'Image_Visual_Quality_norm',
      'Privacy_Body_Exposure_norm',
      'Privacy_Face_Exposure_norm',
      'Privacy_Context_Exposure_norm',
      'I_PA',
  ]

  # Ensure all feature columns exist in DataFrame
  for c in feature_cols:
    if c not in res.columns:
      res[c] = 0.0

  X = res[feature_cols].values

  # Weight Vector (W): Shape (8,)
  # Dimension totals: Text=.20, Image=.25, Privacy=.35, Interaction=.20
  # (manuscript Table A5-1 / Sec. 4.5.2).
  weights_vector = np.array(
      [0.10, 0.10, 0.15, 0.10, 0.04, 0.23, 0.08, 0.20], dtype=np.float64
  )
  assert abs(weights_vector.sum() - 1.0) < 1e-8, 'CMIR weights must sum to 1.0'

  # Dot Product: CMIR = X · W
  res['CMIR'] = np.dot(X, weights_vector)

  # 5. Risk Stratification Routing Logic
  conditions = [
      (res['CMIR'] < 0.155),
      (res['CMIR'] >= 0.155) & (res['CMIR'] < 0.192),
      (res['CMIR'] >= 0.192) & (res['CMIR'] < 0.231),
      (res['CMIR'] >= 0.231),
  ]
  zones = ['Low Risk Zone', 'Gold Zone', 'Medium Risk Zone', 'Masking Zone']
  actions = [
      'Pass (Immediate)',
      'Pass (Standard)',
      'AI Screening',
      'Mask (Automated)',
  ]

  res['Risk_Zone'] = np.select(conditions, zones, default='Masking Zone')
  res['Routing_Action'] = np.select(conditions, actions, default='Mask')

  return res
