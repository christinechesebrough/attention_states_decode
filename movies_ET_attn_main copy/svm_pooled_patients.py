#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 23:06:47 2026

@author: christine
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score
)
from sklearn.svm import LinearSVC
import os, re, sys
import warnings

warnings.filterwarnings(
    "ignore",
    message="y_pred contains classes not in y_true"
)

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="sklearn.metrics"
)

# ==========================================================
## similar to LDA classifer version 3 final

# Repeated Cross Validation
#
# 100 repetitions
# ×
# 5-fold grouped cross validation
#
# = 500 temporary linear svc models
#
# Each temporary model:
#   1. trains on 4 folds
#   2. tests on the remaining fold
#   3. produces one coefficient for every feature
#
# At the end:
#   - average the 500 coefficients for each feature
#   - compute the coefficient standard deviation
#   - compute mean cross-validation accuracy
# ==========================================================

## 5 cv validation done 100x as one split may be unlucky/lucky so 
## trial groups are reshuffled adn 5-fold process is done 100x

#input_file = ("/home/christine/Downloads/final_power_dataframe (1).csv")
atlas = "aparc_aseg" #"Desikan_Killiany"
subset_by_area = False

input_file = f"/media/christine/Samsung/Movie_data/final_power_dataframes/combined_{atlas}_power_dataframe.csv"

original_target_col = "attn_label"
binary_target_col = "attention_class"

attn_scale = "extremes"

if attn_scale == "extremes":
    attn_scale_ext = [0, 1]
    attn_scale_int = [8, 9]
elif attn_scale == "thirds":
    attn_scale_ext = [0, 3]
    attn_scale_int = [6, 9]
elif attn_scale == "full_scale":
    attn_scale_ext = [0, 4]
    attn_scale_int = [5, 9]

random_state = 42

df_all = pd.read_csv(input_file)

print("Original dataframe shape:", df_all.shape)
print("\nOriginal columns:")
print(df_all.columns.tolist())

if subset_by_area:
    atlas_areas = df_all[atlas].dropna().unique()
else:
    atlas_areas = ["All"]

print(f"\n{atlas} areas:")
print(atlas_areas)

atlas_model_results = []
atlas_coefficient_results = []

# ==========================================================
# Run model
# ==========================================================

for area in atlas_areas:

    print("\n" + "=" * 60)

    if subset_by_area:
        print(f"Running model for {atlas}: {area}")
        df = df_all[df_all[atlas] == area].copy()
    else:
        print(f"Running model using all {atlas} areas")
        df = df_all.copy()

    print("=" * 60)

    df = df.dropna(subset=["subject", "recording", "trial", "time", "power", "attn_label", "atlas_freq"]).copy()

    df["attention_class"] = np.select([df["attn_label"].between(*attn_scale_int), df["attn_label"].between(*attn_scale_ext)], [0, 1], default=np.nan)

    df = df.dropna(subset=["attention_class"]).copy()
    df["attention_class"] = df["attention_class"].astype(int)

    print(df[["attn_label", "attention_class"]].drop_duplicates())
    print("Subjects contributing:", df["subject"].unique())
    print("Number of contributing subjects:", df["subject"].nunique())
    
    sample_columns = ["subject", "recording", "trial", "time"]
    
    X_df = df.pivot_table(index=sample_columns, columns="atlas_freq", values="power", aggfunc="mean")
    
    print("Samples:", X_df.shape[0])
    print("Atlas-frequency features:", X_df.shape[1])
    print("Features:", X_df.columns.tolist())
    
    if X_df.isna().any().any():
        dropped_features = X_df.columns[X_df.isna().any()].tolist()
        print("Features excluded for incomplete coverage:", dropped_features)
        X_df = X_df.dropna(axis=1)
    
    y_series = df.groupby(sample_columns)["attention_class"].first().reindex(X_df.index)
    
    sample_info = X_df.index.to_frame(index=False)
    
    groups = (
        sample_info["subject"].astype(str)
        + "_"
        + sample_info["recording"].astype(str)
        + "_trial_"
        + sample_info["trial"].astype(str)
    ).to_numpy()
    
    X = X_df.to_numpy()
    y = y_series.to_numpy()
    feature_names = X_df.columns.to_numpy()
    
    trial_labels = pd.DataFrame({"group": groups, "label": y}).drop_duplicates("group")
    
    print(trial_labels["label"].value_counts().sort_index().rename(index={0: "Internal", 1: "External"}))
        
    ## scaler and svm are refitted every training fold. 
    svm_pipeline = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "svm",
            LinearSVC(
                class_weight="balanced",
                random_state=random_state,
                max_iter=10000
            )
        )
    ])
    
    
    ## train the 500 temporary SVCs
    ## measures the 500 weights per feature
    ## gives accurancy, balanced accuracy, and weights 
    ## calculates mean weight, and std
    
    number_of_repetitions = 100
    number_of_folds = 5
    
    weight_results = []
    accuracy_results = []
    balanced_accuracy_results = []
    
    
    for repetition in range(number_of_repetitions):
    
        cv = StratifiedGroupKFold(
            n_splits=number_of_folds,
            shuffle=True,
            random_state=repetition
        )
    
        for train_index, test_index in cv.split(
            X,
            y,
            groups=groups
        ):
    
            # Create a new temporary scaler and svm
            model = clone(svm_pipeline)
    
            # Train on four folds
            model.fit(
                X[train_index],
                y[train_index]
            )
    
            # Predict the hidden fifth fold
            predictions = model.predict(
                X[test_index]
            )
    
            # Save this model's performance
            accuracy_results.append(
                accuracy_score(
                    y[test_index],
                    predictions
                )
            )
    
            balanced_accuracy_results.append(
                balanced_accuracy_score(
                    y[test_index],
                    predictions
                )
            )
    
            # Retrieve one weight for every feature
            coefficients = (
                model.named_steps["svm"]
                .coef_[0]
            )
    
            weight_results.append(
                pd.DataFrame({
                    "feature": feature_names,
                    "coefficient": coefficients
                })
            )
    
    
    print(
        "Temporary SVM models trained:",
        len(weight_results)
    )
    
    ## average each features 500 weights
    all_weights_df = pd.concat(
        weight_results,
        ignore_index=True
    )
    
    results_df = (
        all_weights_df
        .groupby("feature")["coefficient"]
        .agg(
            mean_coefficient="mean",
            coefficient_std="std"
        )
        .reset_index()
    )
    
    results_df["absolute_coefficient"] = results_df["mean_coefficient"].abs()
    
    print(results_df.head())
    
    feature_parts = results_df["feature"].str.rsplit("_", n=1, expand=True)
    
    results_df[atlas] = feature_parts[0]
    results_df["frequency"] = feature_parts[1]
    
    results_df["direction"] = np.select(
        [
            results_df["mean_coefficient"] > 0,
            results_df["mean_coefficient"] < 0
        ],
        [
            "External",
            "Internal"
        ],
        default="No direction"
    )
    
    ## overall cross-validation performance 
    mean_cv_accuracy = np.mean(accuracy_results)
    
    mean_cv_balanced_accuracy = np.mean(balanced_accuracy_results)
    
    atlas_model_results.append({
        "atlas": atlas,
        "area": area,
        "n_subjects": df["subject"].nunique(),
        "mean_n_contacts": df["n_contacts"].mean(),
        "min_n_contacts": df["n_contacts"].min(),
        "max_n_contacts": df["n_contacts"].max(),
        "n_features": X.shape[1],
        "n_samples": X.shape[0],
        "n_trials": len(np.unique(groups)),
        "mean_cv_accuracy": mean_cv_accuracy,
        "mean_cv_balanced_accuracy": mean_cv_balanced_accuracy
    })
    
    print(
        "Mean CV accuracy:",
        mean_cv_accuracy
    )
    
    print(
        "Mean CV balanced accuracy:",
        mean_cv_balanced_accuracy
    )
    
    ## add to results table
    results_df["mean_cv_accuracy"] = (
        mean_cv_accuracy
    )
    
    results_df[
        "mean_cv_balanced_accuracy"
    ] = mean_cv_balanced_accuracy
    
    results_df = results_df[
        [
            "feature",
            atlas,
            "frequency",
            "direction",
            "mean_coefficient",
            "absolute_coefficient",
            "coefficient_std",
            "mean_cv_accuracy",
            "mean_cv_balanced_accuracy"
        ]
    ]
    
    results_df = (
        results_df
        .sort_values(
            by="mean_coefficient",
            key=lambda column: column.abs(),
            ascending=False
        )
        .reset_index(drop=True)
    )
    
    print(
        results_df
        .head(20)
        .to_string(index=False)
    )
    
        
    safe_area = str(area).replace("/", "-").replace(" ", "_")
    area = safe_area    
    
    
    output_folder = Path(
       f"/media/christine/Samsung/attn_states_decoding/outputs/{atlas}"
    )
    
    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )
    
    output_file = (
        output_folder
        / f"SVM_contact_attention_results_pooled_{attn_scale}_{atlas}_{area}.csv"
    )
    
    results_df.to_csv(
        output_file,
        index=False
    )
    
    print("Saved results to:")
    print(output_file)
    
    ## training one final SVM on all the data
    final_svm_model = clone(svm_pipeline)
    final_svm_model.fit(X,y)
    
    train_predictions = final_svm_model.predict(X)
    train_accuracy = accuracy_score(y, train_predictions)
    
    train_balanced_accuracy = balanced_accuracy_score(y, train_predictions)
    
    print("Training accuracy:", train_accuracy)
    print("Training balanced accuracy:", train_balanced_accuracy)
    
    
    """ to test on new unseen data
    in new notebook, load csv, build feature matrix with same kind of rows and pivot into same way
    (new feature matrix must have same columns and order as training matrix)
    Use Use final_svm_model.predict()
    save predictions 
    """

     # ==========================================================
    # Mean coefficients by atlas grouping
    # ==========================================================
    
    grouping_atlas = atlas
        
    area_frequency_mean_coefficients = results_df[
        [atlas, "frequency", "mean_coefficient", "coefficient_std", "direction"]
    ].copy()
    
    area_frequency_mean_coefficients.insert(0, "model_area", area)
    area_frequency_mean_coefficients.insert(0, "atlas", atlas)
    
    atlas_coefficient_results.append(area_frequency_mean_coefficients)

    # ==========================================================
    # Combined coefficient summary across atlas-area models
    # ==========================================================
    
    atlas_coefficient_results_df = pd.concat(atlas_coefficient_results, ignore_index=True)
    
    atlas_coefficient_results_df = atlas_coefficient_results_df.sort_values(["model_area", "frequency"]).reset_index(drop=True)
    
    atlas_coefficient_results_df["direction"] = np.select(
        [
            atlas_coefficient_results_df["mean_coefficient"] > 0,
            atlas_coefficient_results_df["mean_coefficient"] < 0
        ],
        [
            "External",
            "Internal"
        ],
        default="No direction"
    )
    
    print(f"\nMean coefficients across all {atlas} area-specific models:")
    print(atlas_coefficient_results_df.to_string(index=False))
    
    atlas_coefficients_file = output_folder / f"pooled_SVM_{atlas}_area_mean_coefficients_{attn_scale}.csv"
    
    atlas_coefficient_results_df.to_csv(atlas_coefficients_file, index=False)
    
    print("\nSaved combined coefficient summary to:")
    print(atlas_coefficients_file)
    
# ==========================================================
# Compare model performance across atlas areas
# ==========================================================

atlas_model_results_df = pd.DataFrame(
    atlas_model_results
)

atlas_model_results_df = (
    atlas_model_results_df
    .sort_values(
        "mean_cv_balanced_accuracy",
        ascending=False
    )
    .reset_index(drop=True)
)

print(f"\nModel performance across {atlas} areas:")
print(
    atlas_model_results_df
    .to_string(index=False)
)

atlas_comparison_file = (
    output_folder
    / f"pooled_SVM_{atlas}_area_comparison_{attn_scale}.csv"
)

atlas_model_results_df.to_csv(
    atlas_comparison_file,
    index=False
)

print("\nSaved atlas comparison to:")
print(atlas_comparison_file)