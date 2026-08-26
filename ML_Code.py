# ==============================================================================
# 0. Essential Library Imports and Initial Settings
# ==============================================================================
import warnings
from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold  # Prevent data leakage by ASIN unit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. Data Loading and Preprocessing
# ==============================================================================
file_path = 'Your File Path'
df_raw = pd.read_csv(file_path)
df_raw.columns = df_raw.columns.str.strip().str.lower()


def mape_calc(y_true, y_pred):
    y_t, y_p = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_t - y_p) / (y_t + 1e-10))) * 100


# Define variables by stage (Model 1 -> Model 2 -> Model 3 structure)
control_vars = [
    'days_since_review',
    'media_count',
    'overall_rating',
    'review_length',
    'review_rating',
    'verified_purchase',
]
text_vars = [
    'ceir_text_incon_text_rating_discrepancy_score',
    'ceir_text_text_specific_detail_claims_count',
]
image_vars = [
    'ceir_image_incon_image_rating_discrepancy_score',
    'ceir_image_incon_visual_quality_score',
]
priv_vars = [
    'privarcy_image_privacy_biometric_body_percent',
    'privarcy_image_privacy_biometric_face_percent',
    'privarcy_image_privacy_context_living_space_percent',
]
amp_vars = ['privacy-amplified visual inconsistency']

model_stages = {
    'Model 1\n(Text)': control_vars + text_vars,
    'Model 2\n(+Visual)': control_vars + text_vars + image_vars,
    'Model 3\n(+Privacy, Interaction)': (
        control_vars + text_vars + image_vars + priv_vars + amp_vars
    ),
}

target = 'log_helpful_count'

# Drop missing values for required variables, asin, and amazon_vine
required_cols = (
    model_stages['Model 3\n(+Privacy, Interaction)']
    + [target, 'amazon_vine', 'asin_id']
)
df_clean = df_raw.dropna(subset=required_cols).copy()

# ==============================================================================
# 2. Model Configuration (Fixed random seed 42 and hyperparameter settings)
# ==============================================================================
model_suite = {
    'GBM': GradientBoostingRegressor(
        random_state=42, n_estimators=200, learning_rate=0.03, max_depth=3
    ),
    'RF': RandomForestRegressor(random_state=42, n_estimators=200, max_depth=10),
    'LGBM': LGBMRegressor(
        random_state=42,
        n_estimators=200,
        learning_rate=0.03,
        reg_alpha=0.5,
        verbose=-1,
    ),
    'XGB': XGBRegressor(
        random_state=42,
        objective='reg:squarederror',
        n_estimators=200,
        learning_rate=0.02,
        reg_lambda=5,
    ),
}

total_results = []

# ==============================================================================
# 3. ASIN Group-wise and Stage-wise 5-Fold Cross-Validation (GroupKFold & Pipeline)
# ==============================================================================
for group_name, group_val in [('Vine', 1), ('Non-Vine', 0)]:
    group_df = df_clean[df_clean['amazon_vine'] == group_val].reset_index(
        drop=True
    )
    print(
        f'📊 Running ASIN-based 5-fold cross-validation for {group_name} group (Seed=42)...'
    )

    for stage_name, feature_set in model_stages.items():
        X = group_df[feature_set]
        y = group_df[target]
        groups = group_df['asin_id']  # Group by product unit to prevent data leakage

        for m_name, model in model_suite.items():
            # Apply GroupKFold to prevent the same ASIN from splitting into both train and test sets
            gkf = GroupKFold(n_splits=5)
            fold_r2, fold_rmse, fold_mape, fold_mae = [], [], [], []

            for train_idx, test_idx in gkf.split(X, y, groups=groups):
                X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
                y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

                # Use Pipeline to fit StandardScaler only within each fold (prevents data leakage)
                pipe = Pipeline(
                    [('scaler', StandardScaler()), ('regressor', model)]
                )

                pipe.fit(X_tr, y_tr)
                preds = pipe.predict(X_te)

                fold_r2.append(r2_score(y_te, preds))
                fold_rmse.append(np.sqrt(mean_squared_error(y_te, preds)))
                fold_mape.append(mape_calc(y_te, preds))
                fold_mae.append(mean_absolute_error(y_te, preds))

            total_results.append({
                'Group': (
                    'Vine (Expert)'
                    if group_name == 'Vine'
                    else 'Non-Vine (General)'
                ),
                'Model Step': stage_name,
                'Model': m_name,
                'R2': np.mean(fold_r2),
                'RMSE': np.mean(fold_rmse),
                'MAPE': np.mean(fold_mape),
                'MAE': np.mean(fold_mae),
            })

perf_df = pd.DataFrame(total_results)

# ==============================================================================
# 4. Final Summary Table Output (Table 5 Standard Format)
# ==============================================================================
summary_table = perf_df.pivot_table(
    index=['Group', 'Model'],
    columns='Model Step',
    values=['R2', 'RMSE', 'MAPE', 'MAE'],
)
summary_table = summary_table.reindex(
    ['R2', 'RMSE', 'MAPE', 'MAE'], axis=1, level=0
)
summary_table = summary_table.reindex(
    list(model_stages.keys()), axis=1, level=1
)

print('\n' + '=' * 95)
print(
    '📊 [Table 5] Predictive Performance Progression Summary (ASIN Grouped & Seed=42)'
)
print('=' * 95)
display(summary_table.round(4))