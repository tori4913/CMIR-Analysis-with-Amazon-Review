"""Create the component-attribution summaries used for the Table 8b audit.

The script reports two descriptive decompositions using the dimension-level,
already weighted CMIR components supplied in a local input file:

1. score-variance attribution: Cov(component, CMIR) / Var(CMIR); and
2. criterion-covariance attribution: Cov(component, criterion) divided by
   the sum of component--criterion covariances.

These quantities describe the composition of the observed score and its
linear association with a selected criterion. They are not causal effects.
No input data are bundled or transmitted.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_COMPONENTS = ["d_rt", "d_ri", "i_privacy", "d_interaction"]


def read_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(path)
    else:
        raise ValueError("--input must be a .csv, .xlsx, or .xls file.")
    frame.columns = frame.columns.astype(str).str.strip().str.lower()
    return frame


def parse_components(value: str) -> list[str]:
    columns = [item.strip().lower() for item in value.split(",") if item.strip()]
    if len(columns) < 2:
        raise ValueError("--components must name at least two comma-separated columns.")
    return columns


def safe_share(numerator: pd.Series, denominator: float) -> pd.Series:
    if np.isclose(denominator, 0.0):
        raise ValueError("Attribution denominator is zero; the decomposition is undefined.")
    return numerator / denominator


def covariances_with(frame: pd.DataFrame, columns: list[str], target: pd.Series) -> pd.Series:
    """Return sample covariances without relying on a pandas DataFrame shortcut."""
    return pd.Series({column: frame[column].cov(target) for column in columns})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="Destination .xlsx file for Table 8b audit summaries.")
    parser.add_argument(
        "--components", default=",".join(DEFAULT_COMPONENTS),
        help="Comma-separated, already weighted CMIR component columns."
    )
    parser.add_argument(
        "--criterion", default=None,
        help="Optional local criterion column for the second covariance attribution."
    )
    parser.add_argument(
        "--total-column", default=None,
        help="Optional stored CMIR total. When omitted, the total is the sum of components."
    )
    args = parser.parse_args()

    components = parse_components(args.components)
    frame = read_frame(args.input)
    required = components + ([args.criterion.lower()] if args.criterion else [])
    if args.total_column:
        required.append(args.total_column.lower())
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError("Input is missing required columns: " + ", ".join(missing))

    total_column = args.total_column.lower() if args.total_column else "cmir_total_reconstructed"
    analysis = frame[components].apply(pd.to_numeric, errors="coerce").dropna().copy()
    analysis[total_column] = (
        pd.to_numeric(frame.loc[analysis.index, total_column], errors="coerce")
        if args.total_column else analysis[components].sum(axis=1)
    )
    analysis = analysis.dropna(subset=[total_column])
    if len(analysis) < 3:
        raise ValueError("At least three complete review-level observations are required.")

    cmir_variance = analysis[total_column].var(ddof=1)
    component_covariances = covariances_with(analysis, components, analysis[total_column])
    score_attribution = pd.DataFrame({
        "Component": components,
        "Mean weighted component": [analysis[column].mean() for column in components],
        "Covariance with CMIR": [component_covariances[column] for column in components],
        "Score variance attribution (%)": (
            safe_share(component_covariances, cmir_variance) * 100
        ).values,
    })

    audit_rows = [{
        "Metric": "N complete reviews", "Value": len(analysis),
    }, {
        "Metric": "CMIR variance", "Value": cmir_variance,
    }, {
        "Metric": "Stored-total check (maximum absolute difference)",
        "Value": (
            (analysis[components].sum(axis=1) - analysis[total_column]).abs().max()
            if args.total_column else 0.0
        ),
    }]

    criterion_attribution = None
    if args.criterion:
        criterion = args.criterion.lower()
        criterion_values = pd.to_numeric(frame.loc[analysis.index, criterion], errors="coerce")
        criterion_frame = analysis.assign(_criterion=criterion_values).dropna(subset=["_criterion"])
        covariances = covariances_with(
            criterion_frame, components, criterion_frame["_criterion"]
        )
        criterion_attribution = pd.DataFrame({
            "Component": components,
            f"Covariance with {criterion}": [covariances[column] for column in components],
            "Criterion covariance attribution (%)": (
                safe_share(covariances, covariances.sum()) * 100
            ).values,
        })
        audit_rows.append({"Metric": "N complete criterion reviews", "Value": len(criterion_frame)})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        score_attribution.round(6).to_excel(writer, sheet_name="Score attribution", index=False)
        if criterion_attribution is not None:
            criterion_attribution.round(6).to_excel(
                writer, sheet_name="Criterion attribution", index=False
            )
        pd.DataFrame(audit_rows).to_excel(writer, sheet_name="Audit", index=False)

    print(score_attribution.round(3).to_string(index=False))
    if criterion_attribution is not None:
        print("\n" + criterion_attribution.round(3).to_string(index=False))
    print(f"Saved Table 8b attribution audit to: {args.output}")


if __name__ == "__main__":
    main()
