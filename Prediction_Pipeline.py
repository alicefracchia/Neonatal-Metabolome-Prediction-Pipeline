import numpy as np
import pandas as pd

import joblib
from joblib import Parallel, delayed, parallel_backend
import os 

import xgboost
from xgboost import XGBClassifier

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV

from sklearn.metrics import roc_auc_score

GLOBAL_SEED = 42
global_rng = np.random.default_rng(GLOBAL_SEED)

clean_df_res = pd.read_csv('residualised_by_birth_day.csv', low_memory=False)

new_par_cd = clean_df_res.loc[clean_df_res['final_diag'] == 1, 'new_par'].unique()
clean_df_res_cd = clean_df_res.loc[clean_df_res['new_par'].isin(new_par_cd)]

new_par_uc = clean_df_res.loc[clean_df_res['final_diag'] == 2, 'new_par'].unique()
clean_df_res_uc = clean_df_res.loc[clean_df_res['new_par'].isin(new_par_uc)]

def CI_bootstrap(scores, n_bootstraps=1500, fixed_seed=GLOBAL_SEED, n_jobs=-1):
    """Parallelized bootstrapped confidence interval (CI) estimation."""
    
    rng = np.random.default_rng(fixed_seed)
    scores = np.array(scores)
    seeds = rng.integers(0, 1e9, size=n_bootstraps)  # independent seeds per job generated using fixed_seed

    def single_bootstrap(seed):
        local_rng = np.random.default_rng(seed)
        return np.mean(local_rng.choice(scores, size=len(scores), replace=True))

    with parallel_backend("loky"):
        boot_means = Parallel(n_jobs=n_jobs)(
            delayed(single_bootstrap)(seed) for seed in seeds
        )

    mean_score = np.mean(scores)
    ci = np.percentile(boot_means, [2.5, 97.5])
    return mean_score, tuple(ci)

def permutation_pvalue(y_true, y_pred, observed_score, metric_func, n_permutations=1000, n_jobs=-1, fixed_seed=GLOBAL_SEED):
    """Fast parallelized permutation p-value computation. Tests performance under label shuffle, not model fitting under null"""
    
    rng = np.random.default_rng(fixed_seed)
    perm_matrix = np.array([rng.permutation(len(y_true)) for _ in range(n_permutations)])

    def single_permutation(i):
        return metric_func(y_true[perm_matrix[i]], y_pred)

    with parallel_backend('loky'):
        perm_vals = Parallel(n_jobs=n_jobs)(
            delayed(single_permutation)(i) for i in range(n_permutations)
        )

    pval = np.mean(np.array(perm_vals) >= observed_score)
    return pval

def get_model(name, scoring='auc', cv=5, n_jobs=-1, random_state=GLOBAL_SEED):
    """Get ML model with or without grid search, always returns (model, param_grid)."""
  
    if name == "xgboost":
        model_params = {
            "eval_metric": scoring,
            "random_state": random_state,
            "n_jobs": -1
        }
        
        grid = {
            'n_estimators': [100, 300, 500],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'reg_alpha': [0, 0.1, 1.0],
            'reg_lambda': [1.0, 5.0, 10.0],
            'min_child_weight': [1, 5, 10],
            'gamma': [0, 1, 5]
        }

        model = XGBClassifier(**model_params)

    return model, grid

def preprocess_data(df, target, cap_age):
    
    mask_valid_age = df['age_at_diagnosis'] < cap_age
    mask_na_age_and_match = df['age_at_diagnosis'].isna() & df['new_par'].isin(df.loc[mask_valid_age, 'new_par'])
    df = df[mask_valid_age | mask_na_age_and_match].reset_index(drop=True)

    # Check minimum requirements
    if df.empty or df[target].nunique() < 2 or len(df) < 15:
        print("Skipping due to insufficient samples or imbalance after initial processing.")
        return None, None, None

    print(f"Sample size after initial processing and capping at age {cap_age}: {len(df)}")

    start_col = df.columns.get_loc('X1')
    end_col = df.columns.get_loc('X1350')
    cols = df.columns[min(start_col, end_col):max(start_col, end_col) + 1]
    X = df[cols].to_numpy()
    y = df[target].to_numpy()

    return df, X, y

def generate_single_run(X, y, run_i, scaler, stratify=None):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=GLOBAL_SEED + run_i, stratify=stratify)

    if (len(y_train) < 5 or len(y_test) < 5 or
        len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2):
        print("Skipping due to insufficient samples or imbalance after train-test split.")
        return None

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return (X_train_scaled, y_train, X_test_scaled, y_test)

def generate_runs(processed_df, X, y, scaler, n_runs, n_jobs=-1):

    new_par_ids = processed_df['new_par'].values
    unique_ids = np.unique(new_par_ids)

    if len(unique_ids) < 2:
        print("Insufficient unique new_par IDs for train-test split.")
        return None

    def generate_group_run(run_i):
        id_train, id_test = train_test_split(unique_ids, test_size=0.3, random_state=GLOBAL_SEED + run_i)
        train_mask = np.isin(new_par_ids, id_train)
        test_mask = np.isin(new_par_ids, id_test)

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return (X_train_scaled, y_train, X_test_scaled, y_test)

    results = Parallel(n_jobs=n_jobs)(
        delayed(generate_group_run)(run_i) for run_i in range(n_runs)
    )
    runs = [res for res in results if res is not None]

    return runs

def run_single_binary_model(X_train, y_train, X_test, y_test, model_name, cv_strategy, run_i):
            
    # Pass scale_pos_weight_value to get_model
    base_model, param_grid = get_model(model_name, random_state=GLOBAL_SEED + run_i)
    
    # Removed fit_params related to scale_pos_weight as it's now handled in get_model
    model = (RandomizedSearchCV(base_model, param_grid, n_iter=10, scoring='roc_auc',
                                 cv=cv_strategy, n_jobs=-1) if param_grid else base_model)

    # No fit_params for scale_pos_weight anymore
    model.fit(X_train, y_train)

    best_model = model.best_estimator_ if hasattr(model, "best_estimator_") else model
    best_params = model.best_params_ if hasattr(model, "best_params_") else {}

    y_pred = best_model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_pred)

    return {
        'auc': auc,
        'y_test': y_test,
        'y_pred': y_pred,
        'best_params': best_params
    }


def run_binary_pipeline(df, target='y', model_name='xgboost', cap_age=None,
                        n_bootstraps=1000, n_permutations=1000, n_runs=30, n_jobs=-1):

    print("📦 Preprocessing data...")
    processed_df, X, y = preprocess_data(df, target, cap_age)

    if processed_df is not None:
        print("🔄 Generating train/test splits...")
        runs = generate_runs(processed_df, X, y, StandardScaler(), n_runs, n_jobs=n_jobs)
        
        print("🤖 Running models in parallel...")
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True)

        results = Parallel(n_jobs=n_jobs)(
            delayed(run_single_binary_model)(X_train, y_train, X_test, y_test, model_name, cv_strategy, i)
            for i, (X_train, y_train, X_test, y_test) in enumerate(runs)
        )

        print("📊 Aggregating results and computing confidence intervals...")
        scores = [r['auc'] for r in results]
        best_params_runs = [r['best_params'] for r in results]
        all_y_true = [r['y_test'] for r in results]
        all_y_pred = [r['y_pred'] for r in results]

        y_true_agg = np.concatenate(all_y_true)
        y_pred_agg = np.concatenate(all_y_pred)

        print("📈 Bootstrapping...")
        boot_score, ci = CI_bootstrap(scores, n_bootstraps)
        
        print("🔁 Permutation testing...")
        pval = permutation_pvalue(y_true_agg, y_pred_agg, roc_auc_score(y_true_agg, y_pred_agg),
                               roc_auc_score, n_permutations)

        print("✅ Pipeline completed.")
        return {
            'model_name': model_name,
            'cap_age': cap_age,
            'metric': 'auc',
            'final_score': boot_score,
            'ci_': ci,
            'pval_overall_score': f"{pval!s}",
            'y_true_overall': y_true_agg,
            'y_pred_overall': y_pred_agg,
            'best_params': best_params_runs if best_params_runs else {},
            'n_valid_runs': len(scores)
        }

results_df = pd.DataFrame()

from tqdm import tqdm

output_dir = "final_results"
os.makedirs(output_dir, exist_ok=True)

cap_ages = np.arange(1, 29)
traits_for_regression = ["y"] 
model_names = ['xgboost']

for t in traits_for_regression:
    
    clean_df_current_trait = clean_df_res_uc.dropna(subset=[t]) 

    print(f"\n--- Running models for trait: {t} ---")
    results_trait = []

    for name in model_names:
        for cap in tqdm(cap_ages, desc="Age:", unit="cap_age"):
            result = run_binary_pipeline(  # run_binary_pipeline
                clean_df_current_trait,
                target=t,
                cap_age=cap,
                model_name=name,
                n_runs=30,
                n_bootstraps=1000,
                n_permutations=1000,
                n_jobs=180,
            )

            if result is not None:
                result['trait'] = t
                result['model_name'] = name
                result['cap_age'] = cap
                results_trait.append(result)
                
                trait_df = pd.DataFrame(results_trait)

                trait_file = os.path.join(output_dir, f"results_{t}_backup_uc.csv")
                trait_df.to_csv(trait_file, index=False)
                print(f"Saved results for trait {t} to {trait_file}")

print("\nFinished running all models.")
