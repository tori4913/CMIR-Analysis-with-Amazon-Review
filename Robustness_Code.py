"""
Robustness_Code.py
===================
Implements the robustness / sensitivity analyses that are reported in the
manuscript (Sec. 5.1, Sec. 4.5.2, Appendix 2, Appendix 3) but had NO
corresponding code in the original repository. Complements Regression_Code.py
(baseline OLS+HC3) and ML_Code.py (predictive performance).

Sections implemented:
  1. Segmented regression / breakpoint analysis of CMIR -> helpfulness
     (manuscript: breakpoint ~CMIR 0.235; Appendix 3, Figure A3).
  2. CMIR weighting robustness: Equal-Weight and data-driven Relative
     Weights Analysis (RWA) alternatives to the Primary weighting, with
     Pearson/rank correlation and Jaccard concordance of risk-zone
     membership (manuscript Table 12).
  3. Monte Carlo perturbation of the composite weights (Tight / Moderate /
     Broad regimes via Dirichlet sampling), reporting the distribution of
     score correlations and zone-membership Jaccard concordance relative to
     the Primary specification (manuscript Table 13; N = 10,000; seed = 42).
  4. Alternative outcome specifications: Negative Binomial and GEE models on
     raw helpful-vote counts (vs. OLS on log-helpfulness), plus an early/late
     temporal-split check (manuscript Appendix 2 / Sec. 5.1).

NOTE: This script expects the same analysis dataframe used by
Regression_Code.py / ML_Code.py, i.e. one row per review with:
  - Y_r (or Helpful_Vote_Count), amazon_vine, asin_id, days_since_review
  - the base_elite control + Text_/Image_/Privacy_ raw indicator columns
  - CMIR (output of CMIR_Construction.build_cmir_dot_product_pipeline)
Update INPUT_PATH below (or pass --input) before running.
"""

import argparse

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler

from CMIR_Construction import safe_min_max

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Same raw-indicator groupings used in CMIR_Construction.py / ML_Code.py.
TEXT_INDICATORS = ["Text_Rating_Incon", "Text_Detail_Deficiency"]
IMAGE_INDICATORS = ["Image_Rating_Incon", "Image_Visual_Quality"]
PRIVACY_INDICATORS = [
    "Privacy_Body_Exposure",
    "Privacy_Face_Exposure",
    "Privacy_Context_Exposure",
]
ALL_RAW_INDICATORS = TEXT_INDICATORS + IMAGE_INDICATORS + PRIVACY_INDICATORS

# Primary (manuscript Table A5-1) dimension weights.
PRIMARY_DIM_WEIGHTS = {"text": 0.20, "image": 0.25, "privacy": 0.35, "interaction": 0.20}


# ---------------------------------------------------------------------------
# 1. Segmented regression / breakpoint analysis
# ---------------------------------------------------------------------------
def segmented_regression_breakpoint(df: pd.DataFrame, x_col="CMIR", y_col="Y_r",
                                     grid_points=200):
    """Grid-search a single breakpoint for a two-piece linear model of
    y ~ x below/above the breakpoint, selecting the breakpoint that
    minimizes SSE (a simple, dependency-free stand-in for formal segmented
    regression / Davies' test)."""
    d = df[[x_col, y_col]].dropna().sort_values(x_col)
    x = d[x_col].values
    y = d[y_col].values

    candidates = np.linspace(np.quantile(x, 0.10), np.quantile(x, 0.90), grid_points)
    best = {"breakpoint": None, "sse": np.inf, "beta_left": None, "beta_right": None}

    for bp in candidates:
        left = x <= bp
        right = ~left
        if left.sum() < 20 or right.sum() < 20:
            continue
        sse = 0.0
        betas = []
        for mask in (left, right):
            X = sm.add_constant(x[mask])
            model = sm.OLS(y[mask], X).fit()
            sse += np.sum((y[mask] - model.predict(X)) ** 2)
            betas.append(model.params[1])
        if sse < best["sse"]:
            best.update(breakpoint=bp, sse=sse, beta_left=betas[0], beta_right=betas[1])

    print("\n" + "=" * 20 + " Segmented Regression Breakpoint " + "=" * 20)
    print(f"Estimated breakpoint (CMIR): {best['breakpoint']:.3f}")
    print(f"Slope below breakpoint:      {best['beta_left']:+.3f}")
    print(f"Slope above breakpoint:      {best['beta_right']:+.3f}")
    print(f"Q3 of {x_col} (reference):   {np.quantile(x, 0.75):.3f}")
    return best


# ---------------------------------------------------------------------------
# 2. CMIR weighting robustness: Equal-Weight and RWA alternatives
# ---------------------------------------------------------------------------
def relative_weights_analysis(X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    """Johnson's (2000) Relative Weights Analysis: orthogonalizes predictors
    via SVD, regresses y on the orthogonal variables, then reallocates
    explained variance back to the original (correlated) predictors.
    Returns non-negative importance weights that sum to 1."""
    Xs = StandardScaler().fit_transform(X.values)
    ys = StandardScaler().fit_transform(y.values.reshape(-1, 1)).ravel()

    U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
    Z = U * S  # orthogonal variables spanning the same column space as Xs
    Z = Z / Z.std(axis=0, keepdims=True)

    lambda_ = np.linalg.lstsq(Z, Xs, rcond=None)[0]  # regress each X_j on Z
    beta = np.linalg.lstsq(Z, ys, rcond=None)[0]      # regress y on Z

    raw_weights = (lambda_ ** 2).T @ (beta ** 2)
    raw_weights = np.clip(raw_weights, 0, None)
    return raw_weights / raw_weights.sum()


def compute_cmir_variant(df: pd.DataFrame, dim_weights: dict) -> pd.Series:
    """Recompute CMIR under an arbitrary dimension-weight dict
    {'text','image','privacy','interaction'} using the same within-dimension
    allocation as CMIR_Construction.py (Primary spec) for text/image, but
    EQUAL within-dimension allocation for privacy (used by the Equal-Weight
    variant) unless overridden."""
    norm = {c: safe_min_max(df[c]) for c in ALL_RAW_INDICATORS if c in df.columns}

    ceir_text = 0.5 * norm["Text_Rating_Incon"] + 0.5 * norm["Text_Detail_Deficiency"]
    ceir_image = 0.6 * norm["Image_Rating_Incon"] + 0.4 * norm["Image_Visual_Quality"]
    i_privacy = (
        (1 / 3) * norm["Privacy_Body_Exposure"]
        + (1 / 3) * norm["Privacy_Face_Exposure"]
        + (1 / 3) * norm["Privacy_Context_Exposure"]
    )
    i_pa = ceir_image * i_privacy

    return (
        dim_weights["text"] * ceir_text
        + dim_weights["image"] * ceir_image
        + dim_weights["privacy"] * i_privacy
        + dim_weights["interaction"] * i_pa
    )


def risk_zones(cmir: pd.Series) -> pd.Series:
    q1, q2, q3 = cmir.quantile([0.25, 0.50, 0.75])
    return pd.cut(
        cmir, bins=[-np.inf, q1, q2, q3, np.inf],
        labels=["Low Risk Zone", "Gold Zone", "Medium Risk Zone", "Masking Zone"],
    )


def jaccard_concordance(zones_a: pd.Series, zones_b: pd.Series) -> float:
    """Mean per-zone Jaccard index between two zone assignments over the
    same observations (manuscript's zone-membership concordance metric)."""
    scores = []
    for z in zones_a.cat.categories:
        a = set(zones_a[zones_a == z].index)
        b = set(zones_b[zones_b == z].index)
        union = a | b
        scores.append(len(a & b) / len(union) if union else 1.0)
    return float(np.mean(scores))


def weighting_robustness(df: pd.DataFrame, y_col="Y_r"):
    print("\n" + "=" * 20 + " CMIR Weighting Robustness " + "=" * 20)

    primary_cmir = df["CMIR"]
    primary_zones = risk_zones(primary_cmir)

    equal_weights = {"text": 0.25, "image": 0.25, "privacy": 0.25, "interaction": 0.25}
    equal_cmir = compute_cmir_variant(df, equal_weights)
    equal_zones = risk_zones(equal_cmir)

    dim_features = pd.DataFrame({
        "text": 0.5 * safe_min_max(df["Text_Rating_Incon"]) + 0.5 * safe_min_max(df["Text_Detail_Deficiency"]),
        "image": 0.6 * safe_min_max(df["Image_Rating_Incon"]) + 0.4 * safe_min_max(df["Image_Visual_Quality"]),
        "privacy": (safe_min_max(df["Privacy_Body_Exposure"]) + safe_min_max(df["Privacy_Face_Exposure"]) + safe_min_max(df["Privacy_Context_Exposure"])) / 3,
    })
    dim_features["interaction"] = dim_features["image"] * dim_features["privacy"]

    rwa_raw = relative_weights_analysis(dim_features, df[y_col])
    rwa_weights = dict(zip(dim_features.columns, rwa_raw))
    rwa_cmir = compute_cmir_variant(df, rwa_weights)
    rwa_zones = risk_zones(rwa_cmir)

    rows = []
    for name, variant_cmir, variant_zones in [
        ("Primary", primary_cmir, primary_zones),
        ("Equal-Weight", equal_cmir, equal_zones),
        ("RWA (data-driven)", rwa_cmir, rwa_zones),
    ]:
        pearson_r = primary_cmir.corr(variant_cmir)
        kendall_tau = primary_cmir.corr(variant_cmir, method="kendall")
        jaccard = jaccard_concordance(primary_zones, variant_zones)
        rows.append({
            "Weighting": name,
            "Weights (T/I/P/A)": {k: round(v, 3) for k, v in
                                   (PRIMARY_DIM_WEIGHTS if name == "Primary"
                                    else equal_weights if name == "Equal-Weight"
                                    else rwa_weights).items()},
            "Pearson r vs Primary": round(pearson_r, 3),
            "Kendall tau vs Primary": round(kendall_tau, 3),
            "Jaccard concordance": round(jaccard, 3),
        })

    result_df = pd.DataFrame(rows)
    print(result_df.to_string(index=False))
    return result_df


# ---------------------------------------------------------------------------
# 3. Monte Carlo weight perturbation
# ---------------------------------------------------------------------------
def monte_carlo_perturbation(df: pd.DataFrame, n_draws=10_000,
                              concentration_regimes=None):
    """Perturb the four dimension weights via Dirichlet sampling centered on
    the Primary specification, at increasing dispersion (Tight/Moderate/
    Broad), and report the distribution of Pearson r and Jaccard
    concordance relative to the Primary CMIR/zones (manuscript Table 13)."""
    if concentration_regimes is None:
        # Higher concentration => tighter draws around the Primary weights.
        concentration_regimes = {"Tight": 400, "Moderate": 100, "Broad": 25}

    primary_cmir = df["CMIR"]
    primary_zones = risk_zones(primary_cmir)
    alpha0 = np.array([
        PRIMARY_DIM_WEIGHTS["text"], PRIMARY_DIM_WEIGHTS["image"],
        PRIMARY_DIM_WEIGHTS["privacy"], PRIMARY_DIM_WEIGHTS["interaction"],
    ])

    print("\n" + "=" * 20 + f" Monte Carlo Perturbation (N={n_draws}, seed={RANDOM_SEED}) " + "=" * 20)
    summary_rows = []
    rng = np.random.default_rng(RANDOM_SEED)

    for regime, concentration in concentration_regimes.items():
        pearsons, jaccards = [], []
        for _ in range(n_draws):
            draw = rng.dirichlet(alpha0 * concentration)
            dw = {"text": draw[0], "image": draw[1], "privacy": draw[2], "interaction": draw[3]}
            variant_cmir = compute_cmir_variant(df, dw)
            pearsons.append(primary_cmir.corr(variant_cmir))
            jaccards.append(jaccard_concordance(primary_zones, risk_zones(variant_cmir)))

        pearsons, jaccards = np.array(pearsons), np.array(jaccards)
        summary_rows.append({
            "Perturbation": regime,
            "Dirichlet concentration": concentration,
            "Median Pearson r": round(np.median(pearsons), 3),
            "Pearson r [5th, 95th pct]": (round(np.quantile(pearsons, .05), 3),
                                          round(np.quantile(pearsons, .95), 3)),
            "Median Jaccard": round(np.median(jaccards), 3),
            "Jaccard [5th, 95th pct]": (round(np.quantile(jaccards, .05), 3),
                                        round(np.quantile(jaccards, .95), 3)),
        })

    result_df = pd.DataFrame(summary_rows)
    print(result_df.to_string(index=False))
    return result_df


# ---------------------------------------------------------------------------
# 4. Alternative outcome specifications + temporal split
# ---------------------------------------------------------------------------
def alternative_outcome_models(df: pd.DataFrame, final_model_vars: list,
                                count_col="Helpful_Vote_Count",
                                cluster_col="asin_id"):
    """Fit Negative Binomial and GEE (exchangeable, clustered by ASIN)
    models on raw helpful-vote counts, alongside the OLS-on-log baseline,
    to check that core associations survive alternative outcome
    specifications (manuscript Appendix 2)."""
    print("\n" + "=" * 20 + " Alternative Outcome Specifications " + "=" * 20)
    d = df.dropna(subset=final_model_vars + [count_col, cluster_col]).copy()
    d[final_model_vars] = StandardScaler().fit_transform(d[final_model_vars])

    formula = f"{count_col} ~ " + " + ".join(final_model_vars)

    # Estimate the NB dispersion (alpha) via statsmodels' discrete NB2 model
    # instead of assuming alpha=1 (the GLM family default), which biases
    # standard errors when the data are over-dispersed relative to Poisson.
    nb_model = smf.negativebinomial(formula=formula, data=d).fit(disp=False)
    print("\n--- Negative Binomial (NB2, dispersion estimated) ---")
    print(nb_model.summary())

    # GEE with a Poisson link can fail to converge (overflow / rank-deficient
    # working matrix) on sparse or near-collinear real review data. Guard it
    # so the rest of the robustness suite still runs and reports the issue
    # explicitly instead of crashing or silently returning NaNs.
    gee_model = None
    try:
        gee_model = smf.gee(
            formula=formula, groups=cluster_col, data=d,
            family=sm.families.Poisson(),
            cov_struct=sm.cov_struct.Exchangeable(),
        ).fit(maxiter=200)
        if gee_model.params.isna().any():
            raise ValueError("GEE converged to NaN coefficients")
        print("\n--- GEE (Poisson, exchangeable, clustered by ASIN) ---")
        print(gee_model.summary())
    except Exception as e:
        print(f"\n[WARN] GEE (Poisson) did not converge cleanly: {e}")
        print("       Consider dropping near-collinear controls or checking")
        print("       for zero-variance columns within an ASIN cluster.")

    return nb_model, gee_model


def temporal_split_check(df: pd.DataFrame, x_col="CMIR", y_col="Y_r",
                          date_col="days_since_review"):
    """Split the sample at the median review age and re-estimate the
    CMIR-helpfulness association within each half, as a simple temporal
    robustness check (manuscript Sec. 5.1)."""
    print("\n" + "=" * 20 + " Temporal Robustness Split " + "=" * 20)
    median_age = df[date_col].median()
    for label, mask in [
        ("Older half (long-elapsed reviews)", df[date_col] >= median_age),
        ("Newer half (recently posted reviews)", df[date_col] < median_age),
    ]:
        sub = df[mask].dropna(subset=[x_col, y_col])
        X = sm.add_constant(sub[[x_col]])
        model = sm.OLS(sub[y_col], X).fit(cov_type="HC3")
        print(f"\n{label} (N={len(sub)}): beta_CMIR = {model.params[x_col]:.4f} "
              f"(p={model.pvalues[x_col]:.4f})")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="your_data_file.csv",
                         help="Analysis dataset with CMIR + raw indicators + Y_r")
    parser.add_argument("--mc-draws", type=int, default=10_000)
    args = parser.parse_args()

    df = pd.read_csv(args.input) if args.input.endswith(".csv") else pd.read_excel(args.input)
    if "Y_r" not in df.columns and "Helpful_Vote_Count" in df.columns:
        df["Y_r"] = np.log1p(df["Helpful_Vote_Count"])

    segmented_regression_breakpoint(df)
    weighting_robustness(df)
    monte_carlo_perturbation(df, n_draws=args.mc_draws)

    base_elite = [
        "days_since_review", "media_count", "overall_rating", "review_length",
        "s_r", "verified_purchase",
    ] + ALL_RAW_INDICATORS
    if "Helpful_Vote_Count" in df.columns and "asin_id" in df.columns:
        alternative_outcome_models(df, base_elite)
    temporal_split_check(df)
