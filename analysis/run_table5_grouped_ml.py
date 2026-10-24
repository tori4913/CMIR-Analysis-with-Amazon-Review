"""Reproduce the grouped machine-learning evaluation reported in Table 5.

The script intentionally contains no review data. Supply a local CSV or XLSX
file with the required columns, and it writes only derived results to the
user-selected output directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


CONTROL_VARS = [
    "days_since_review", "media_count", "overall_rating", "review_length",
    "review_rating", "verified_purchase",
]
TEXT_VARS = [
    "ceir_text_incon_text_rating_discrepancy_score",
    "ceir_text_text_specific_detail_claims_count",
]
IMAGE_VARS = [
    "ceir_image_incon_image_rating_discrepancy_score",
    "ceir_image_incon_visual_quality_score",
]
PRIVACY_AND_INTERACTION_VARS = [
    "privarcy_image_privacy_biometric_body_percent",
    "privarcy_image_privacy_biometric_face_percent",
    "privarcy_image_privacy_context_living_space_percent",
    "privacy-amplified visual inconsistency",
]


def read_frame(path: Path) -> pd.DataFrame:
    """Read a CSV or Excel file and normalize column-name whitespace/case."""
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        raise ValueError("--input must be a .csv, .xlsx, or .xls file.")
    df.columns = df.columns.astype(str).str.strip().str.lower()
    return df


def validate_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError("Input is missing required columns: " + ", ".join(missing))


def build_models(seed: int) -> dict[str, object]:
    return {
        "GBM": GradientBoostingRegressor(
            random_state=seed, n_estimators=200, learning_rate=0.03, max_depth=3
        ),
        "Random Forest": RandomForestRegressor(
            random_state=seed, n_estimators=200, max_depth=10, n_jobs=-1
        ),
        "LGBM": LGBMRegressor(
            random_state=seed, n_estimators=200, learning_rate=0.03,
            reg_alpha=0.5, verbose=-1
        ),
        "XGBoost": XGBRegressor(
            random_state=seed, objective="reg:squarederror", n_estimators=200,
            learning_rate=0.02, reg_lambda=5, n_jobs=-1
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    stages = {
        "Model 1 (Text)": CONTROL_VARS + TEXT_VARS,
        "Model 2 (+Image)": CONTROL_VARS + TEXT_VARS + IMAGE_VARS,
        "Model 3 (+Privacy, Interaction)": (
            CONTROL_VARS + TEXT_VARS + IMAGE_VARS + PRIVACY_AND_INTERACTION_VARS
        ),
    }
    required = stages["Model 3 (+Privacy, Interaction)"] + [
        "log_helpful_count", "amazon_vine", "asin_id"
    ]
    df = read_frame(args.input)
    validate_columns(df, required)
    df = df.dropna(subset=required).copy()
    if df.empty:
        raise ValueError("No complete rows remain after applying the required-column filter.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    models = build_models(args.seed)

    for group_name, group_value in (("Vine", 1), ("Non-Vine", 0)):
        group = df.loc[df["amazon_vine"] == group_value].reset_index(drop=True)
        if group.empty:
            raise ValueError(f"No rows were found for {group_name} reviews.")
        if group["asin_id"].nunique() < args.folds:
            raise ValueError(
                f"{group_name} has fewer than {args.folds} unique ASINs; reduce --folds."
            )
        for stage_name, features in stages.items():
            x, y, groups = group[features], group["log_helpful_count"], group["asin_id"]
            splitter = GroupKFold(n_splits=args.folds)
            for model_name, model in models.items():
                for fold, (train_index, test_index) in enumerate(
                    splitter.split(x, y, groups), start=1
                ):
                    pipeline = Pipeline([
                        ("scaler", StandardScaler()), ("regressor", model),
                    ])
                    pipeline.fit(x.iloc[train_index], y.iloc[train_index])
                    prediction = pipeline.predict(x.iloc[test_index])
                    rows.append({
                        "Group": group_name,
                        "Model": model_name,
                        "Stage": stage_name,
                        "Fold": fold,
                        "Train_N": len(train_index),
                        "Test_N": len(test_index),
                        "Train_ASIN": int(groups.iloc[train_index].nunique()),
                        "Test_ASIN": int(groups.iloc[test_index].nunique()),
                        "R2": r2_score(y.iloc[test_index], prediction),
                        "RMSE": float(np.sqrt(mean_squared_error(y.iloc[test_index], prediction))),
                        "MAE": mean_absolute_error(y.iloc[test_index], prediction),
                    })

    fold_results = pd.DataFrame(rows)
    summary = fold_results.groupby(["Group", "Model", "Stage"], as_index=False).agg(
        R2=("R2", "mean"), RMSE=("RMSE", "mean"), RMSE_SD=("RMSE", "std"),
        MAE=("MAE", "mean"),
    )
    fold_results.to_csv(args.output_dir / "table5_grouped_ml_folds.csv", index=False)
    summary.round(6).to_csv(args.output_dir / "table5_grouped_ml_summary.csv", index=False)
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
