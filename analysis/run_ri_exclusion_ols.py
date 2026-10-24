"""Run the Rating-Image Inconsistency (RI) exclusion analysis for Reviewer 7-1.

This script retains the group-specific, three-stage OLS specification while
removing RI from Stages 2 and 3. It does not bundle or transmit any data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler


CONTROLS = [
    "days_since_review", "media_count", "overall_rating", "review_length",
    "review_rating", "verified_purchase",
]
TEXT_VARS = ["text_rating_incon", "text_detail_deficiency"]
IMAGE_QUALITY = "image_visual_quality"
PRIVACY_AND_INTERACTION = [
    "privacy_face_exposure", "privacy_body_exposure", "privacy_living_context",
    "privacy_visual_amplified",
]


def read_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        raise ValueError("--input must be a .csv, .xlsx, or .xls file.")
    df.columns = df.columns.astype(str).str.strip().str.lower()
    return df


def fit_hc3_ols(df: pd.DataFrame, outcome: str, features: list[str]) -> tuple[pd.DataFrame, dict[str, float]]:
    model_df = df[[outcome] + features].dropna().copy()
    x = StandardScaler().fit_transform(model_df[features])
    x = sm.add_constant(pd.DataFrame(x, columns=features, index=model_df.index))
    fit = sm.OLS(model_df[outcome], x).fit(cov_type="HC3")
    coefficients = pd.DataFrame({
        "Variable": fit.params.index,
        "Coefficient": fit.params.values,
        "HC3_SE": fit.bse.values,
        "t": fit.tvalues.values,
        "p": fit.pvalues.values,
    })
    model_fit = {
        "N": int(fit.nobs), "R2": float(fit.rsquared),
        "Adjusted_R2": float(fit.rsquared_adj), "F": float(fit.fvalue),
        "F_p": float(fit.f_pvalue),
    }
    return coefficients, model_fit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="Destination .xlsx file for coefficient and fit sheets.")
    parser.add_argument("--outcome", default="helpful_vote_log")
    parser.add_argument("--vine-column", default="amazon_vine")
    args = parser.parse_args()

    df = read_frame(args.input)
    stages = {
        "Model 1 (Text)": CONTROLS + TEXT_VARS,
        "Model 2 (+Image, RI excluded)": CONTROLS + TEXT_VARS + [IMAGE_QUALITY],
        "Model 3 (+Privacy, Interaction; RI excluded)": (
            CONTROLS + TEXT_VARS + [IMAGE_QUALITY] + PRIVACY_AND_INTERACTION
        ),
    }
    required = [args.outcome, args.vine_column] + stages["Model 3 (+Privacy, Interaction; RI excluded)"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError("Input is missing required columns: " + ", ".join(missing))

    all_coefficients: list[pd.DataFrame] = []
    all_fit: list[dict[str, object]] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        for group_name, group_value in (("Vine", 1), ("Non-Vine", 0), ("All", None)):
            group_df = df if group_value is None else df.loc[df[args.vine_column] == group_value]
            coefficients_for_group = []
            for stage_name, features in stages.items():
                coefficients, model_fit = fit_hc3_ols(group_df, args.outcome, features)
                coefficients.insert(0, "Model", stage_name)
                coefficients_for_group.append(coefficients)
                all_coefficients.append(coefficients.assign(Group=group_name))
                all_fit.append({"Group": group_name, "Model": stage_name, **model_fit})
            pd.concat(coefficients_for_group, ignore_index=True).to_excel(
                writer, sheet_name=f"RI-excl {group_name} OLS"[:31], index=False
            )
        pd.DataFrame(all_fit).to_excel(writer, sheet_name="OLS model fit", index=False)
        pd.concat(all_coefficients, ignore_index=True).to_excel(
            writer, sheet_name="OLS all coefficients", index=False
        )

    print(f"Saved RI-exclusion OLS results to: {args.output}")


if __name__ == "__main__":
    main()
