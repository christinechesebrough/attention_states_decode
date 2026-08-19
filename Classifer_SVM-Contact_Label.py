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

patient_id = 'LH019'
input_file = (f"/media/christine/Samsung/Movie_data/final_power_dataframes/{patient_id}_final_power_dataframe.csv")

subset_by_area = False

model_name = "SVM"

original_target_col = "attn_label"
binary_target_col = "attention_class"

attn_scale = 'extremes'

if attn_scale == 'extremes':
    attn_scale_ext = [0, 1]
    attn_scale_int = [8, 9]
elif attn_scale == 'thirds':
    attn_scale_ext = [0, 3]
    attn_scale_int = [6, 9]
elif attn_scale == 'full_scale':
    attn_scale_ext = [0, 4]
    attn_scale_int = [5, 9]
    
atlas = "All"

random_state = 42

df_all = pd.read_csv(input_file)

print("Original dataframe shape:", df_all.shape)
print("\nOriginal columns:")
print(df_all.columns.tolist())


# ==========================================================
# Run model separately for each atlas area
# ==========================================================

if subset_by_area:
    # Get all unique atlas labels
    atlas_areas = (
        df_all[atlas]
        .dropna()
        .loc[lambda x: x != "Out"]
        .loc[lambda x: x != 'FreeSurfer_Defined_Medial_Wall']
        .unique()
    )
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
    print("Filtered dataframe shape:", df.shape)

    # ------------------------------------------------------
    # Continue with existing pipeline
    # ------------------------------------------------------

        
    ##remove any NaN
    df = df.dropna(
        subset=[
            "subject",
            "recording",
            "trial",
            "time",
            "power",
            "attn_label",
            "valence",
            "pow_contact_freq"
        ]
    ).copy()
    
    df = df[df['Desikan_Killiany'] != "Out"] # remove contacts labeled as "Out" in the Desikan Killiany atlas
    
    ##select from the label any values between 1,4 and 6,10 and set as 0, 1 respectiviely and create new column in df
    ## 0 = Internal
    ## 1 = External
    
    df["attention_class"] = np.select([df["attn_label"].between(*attn_scale_int),df["attn_label"].between(*attn_scale_ext)],[0,1],default=np.nan)
    
    
    df = df.dropna(
        subset=["attention_class"]
    ).copy()
    
    ## set as int 
    df["attention_class"] = (
        df["attention_class"]
        .astype(int))
    
    print(df[["attn_label", "attention_class"]].drop_duplicates())
    
    
    ## create contact-freq matrix 
    
    sample_columns = [
        "subject",
        "recording",
        "trial",
        "time"
    ]
    
    X_df = df.pivot_table(
        index=sample_columns,
        columns="pow_contact_freq",
        values="power",
        aggfunc="mean"
    )
    
    print("Samples:", X_df.shape[0])
    print("Contact-frequency features:", X_df.shape[1])
    
    print("Missing values in X_df:", X_df.isna().sum().sum())
    
    if X_df.isna().any().any():
        missing_features = X_df.isna().sum()
        print(
            missing_features[missing_features > 0]
            .sort_values(ascending=False)
        )
    
    test_df = X_df.dropna(axis = 1)
    X_df = test_df
        
    ## creates trials and groups 
    y_series = (
        df.groupby(sample_columns)["attention_class"]
        .first()
        .reindex(X_df.index)
    )
    
    sample_info = X_df.index.to_frame(
        index=False
    )
    
    groups = (
        sample_info["subject"].astype(str)
        + "_"
        + sample_info["recording"].astype(str)
        + "_trial_"
        + sample_info["trial"].astype(str)
    ).to_numpy()
    
    X = X_df.to_numpy()
    print(len(X))
    
    y = y_series.to_numpy()
    
    feature_names = X_df.columns.to_numpy()
    
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Number of trial groups:", len(np.unique(groups)))
    
    print("NaNs in X:", np.isnan(X).sum())
    print("NaNs in y:", np.isnan(y).sum())
    print("Missing groups:", pd.isna(groups).sum())
    
    assert not np.isnan(X).any(), "X contains missing values"
    assert not np.isnan(y).any(), "y contains missing values"
    assert not pd.isna(groups).any(), "groups contains missing values"
    
    print("No missing values in final SVM inputs.")
    
    ## count independent trials
    trial_labels = pd.DataFrame({
        "group": groups,
        "label": y
    }).drop_duplicates(
        subset="group"
    )
    
    print(
        trial_labels["label"]
        .value_counts()
        .sort_index()
        .rename(
            index={
                1: "External",
                0: "Internal"
            }
        )
    )
    
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
    
    print(results_df.head())
    
    results_df["absolute_coefficient"] = results_df["mean_coefficient"].abs()
    
    ## seperate contact and frequency 
    feature_parts = (
        results_df["feature"]
        .str.rsplit(
            "_",
            n=1,
            expand=True
        )
    )
    
    results_df["contact"] = feature_parts[0]
    results_df["frequency"] = feature_parts[1]
    
    metadata_cols = [
        "Yeo7",
        "Yeo17",
        "Desikan_Killiany",
        "aparc_aseg",
        "LEPTO_coords_1",
        "LEPTO_coords_2",
        "LEPTO_coords_3",
        'PIALVOX_coords_1', 'PIALVOX_coords_2', 'PIALVOX_coords_3',
             'fsaverage_coords_1', 'fsaverage_coords_2', 'fsaverage_coords_3'
        ]
    
    
    column_lookup = {col.lower(): col for col in df.columns}

    rename_map = {
        column_lookup[col.lower()]: col
        for col in metadata_cols
        if col.lower() in column_lookup
    }
    
    df = df.rename(columns=rename_map)
    
    contact_metadata = df[["channel"] + metadata_cols].drop_duplicates(subset="channel").copy()
    
    contact_metadata = contact_metadata.rename(columns={"channel": "contact"})
    
    results_df = results_df.merge(contact_metadata, on="contact", how="left", validate="many_to_one")
        
    # contact_metadata = df[["channel"] + metadata_cols].drop_duplicates(subset="channel").copy()

    # contact_metadata = contact_metadata.rename(columns={"channel": "contact"})
    
    # results_df = results_df.merge(contact_metadata, on="contact", how="left", validate="many_to_one")
    
    ## adds external/internal "direction" as one single contact isnt enough to link  
    ## Positive mean coefficient → External direction
    ## Negative mean coefficient → Internal direction
    
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
        "subject": patient_id,
        "atlas": atlas,
        "area": area,
        "n_channels": df["channel"].nunique(),
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
            "contact",
            "frequency",
            "direction",
            "mean_coefficient",
            "absolute_coefficient",
            "coefficient_std",
            "mean_cv_accuracy",
            "mean_cv_balanced_accuracy",
            "Yeo7",
            "Yeo17",
            "Desikan_Killiany",
            "aparc_aseg",
            "LEPTO_coords_1",
            "LEPTO_coords_2",
            "LEPTO_coords_3",
            'PIALVOX_coords_1', 'PIALVOX_coords_2', 'PIALVOX_coords_3',
             'fsaverage_coords_1', 'fsaverage_coords_2', 'fsaverage_coords_3'
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
    
    output_folder = Path(
       f"/media/christine/Samsung/attn_states_decoding/outputs/{patient_id}"
    )
    
    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    output_file = (
        output_folder
        / f"{patient_id}_SVM_contact_attention_results_{attn_scale}_{atlas}_{area}.csv"
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

#      # ==========================================================
#     # Mean coefficients by atlas grouping
#     # ==========================================================
    
    grouping_atlas = atlas
    
    if atlas != 'All':
        area_frequency_mean_coefficients = (
            results_df
            .groupby(
                [grouping_atlas, "frequency"],
                dropna=False
            )["mean_coefficient"]
            .agg(
                mean_coefficient="mean",
                mean_abs_coefficient=lambda x: x.abs().mean(),
                coefficient_std="std",
                n_features="count"
            )
            .reset_index()
        )
    elif atlas == 'All':
        area_frequency_mean_coefficients = (
            results_df
            .groupby(
                ["frequency"],
                dropna=False
            )["mean_coefficient"]
            .agg(
                mean_coefficient="mean",
                mean_abs_coefficient=lambda x: x.abs().mean(),
                coefficient_std="std",
                n_features="count"
            )
            .reset_index()
        )
    
    #area_frequency_mean_coefficients = results_df.groupby("frequency")["mean_coefficient"].agg(mean_coefficient="mean", mean_abs_coefficient=lambda x: x.abs().mean(), coefficient_std="std", n_features="count").reset_index()
    area_frequency_mean_coefficients.insert(0, "model_area", area)
    area_frequency_mean_coefficients.insert(0, "atlas", atlas)
    area_frequency_mean_coefficients.insert(0, "subject", patient_id)
    
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
    
    atlas_coefficients_file = output_folder / f"{patient_id}_SVM_{atlas}_area_mean_coefficients_{attn_scale}.csv"
    
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

if subset_by_area:
    print(f"\nModel performance across {atlas} areas:")
else:
    print(f"\nModel performance using all {atlas} areas:")
    
atlas_comparison_file = (
    output_folder
    / f"{patient_id}_SVM_{atlas}_area_comparison_{attn_scale}.csv"
)

atlas_model_results_df.to_csv(
    atlas_comparison_file,
    index=False
)

print("\nSaved atlas comparison to:")
print(atlas_comparison_file)

#%%
import matplotlib.pyplot as plt
# plot outcomes

freq_order = ["theta","alpha","beta","gamma","HFA"]

if subset_by_area:
    
    #coefficients across atlases and freqs
    coef_df = atlas_coefficient_results_df
    
    # coefficients across atlases overall
    perf_df = atlas_model_results_df
    
    if atlas == "Yeo7":
        area_order = ['Default', 'Dorsal_Attention', 'Frontoparietal', 'Limbic',
               'Somatomotor', 'Ventral_Attention', 'Visual']
    
    outdir = f"/media/christine/Samsung/attn_states_decoding/{patient_id}_{model_name}_{atlas}_bargraphs"
    os.makedirs(outdir, exist_ok=True)
    
    # -----------------------------
    # 1. Signed mean coefficients
    # -----------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(area_order))
    width = 0.15
    
    for i, freq in enumerate(freq_order):
        d = coef_df[coef_df["frequency"] == freq].set_index("area").loc[area_order]
        xpos = x + (i - (len(freq_order)-1)/2) * width
        ax.bar(xpos, d["mean_coefficient"], width=width, label=freq)
      #  ax.errorbar(xpos, d["mean_coefficient"], yerr=d["coefficient_std"], fmt="none", capsize=2, linewidth=0.8)
    
    ax.axhline(0, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(area_order, rotation=35, ha="right")
    ax.set_ylabel(f"Mean {model_name} coefficient")
    ax.set_xlabel(f"{atlas}")
    ax.set_title(f"{patient_id}: Mean {model_name} coefficients across {atlas} area-specific models ({attn_scale})")
    ax.legend(title="Frequency", frameon=False, ncol=5)
    fig.tight_layout()
    
    signed_png = os.path.join(outdir,f"{patient_id}_{model_name}_{atlas}_mean_coefficients_{attn_scale}.png")
    fig.savefig(signed_png, dpi=300, bbox_inches="tight")
    
    # -----------------------------
    # 2. Mean absolute coefficients
    # -----------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, freq in enumerate(freq_order):
        d = coef_df[coef_df["frequency"] == freq].set_index("area").loc[area_order]
        xpos = x + (i - (len(freq_order)-1)/2) * width
        ax.bar(xpos, d["mean_abs_coefficient"], width=width, label=freq)
    
    ax.set_xticks(x)
    ax.set_xticklabels(area_order, rotation=35, ha="right")
    ax.set_ylabel(f"Mean absolute {model_name} coefficient")
    ax.set_xlabel(f"{atlas} parcels")
    ax.set_title(f"{patient_id}: Mean absolute {model_name} coefficients across {atlas} area-specific models ({attn_scale})")
    ax.legend(title="Frequency", frameon=False, ncol=5)
    fig.tight_layout()
    
    abs_png = os.path.join(outdir,f"{patient_id}_{model_name}_{atlas}_mean_absolute_coefficients_{attn_scale}.png")
    fig.savefig(abs_png, dpi=300, bbox_inches="tight")
    plt.show()
    
    # -----------------------------
    # 3. Model performance
    # -----------------------------
    perf_plot = perf_df[
        (perf_df["subject"] == patient_id) &
        (perf_df["atlas"] == atlas)
    ].copy()
    
    perf_plot["mean_cv_accuracy"] = pd.to_numeric(perf_plot["mean_cv_accuracy"], errors="coerce")
    perf_plot = perf_plot.dropna(subset=["mean_cv_accuracy"]).sort_values("mean_cv_accuracy", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(perf_plot["area"], perf_plot["mean_cv_accuracy"])
    
    ax.axhline(0.5, linestyle="--", linewidth=1)
    ax.set_ylim(0.45, 0.95)
    ax.set_ylabel("Mean cross-validated accuracy")
    ax.set_xlabel(f"{atlas} parcels")
    ax.set_title(f"{patient_id}: {model_name} decoding performance across {atlas} parcels ({attn_scale})")
    ax.tick_params(axis="x", rotation=35)
    
    for bar, value in zip(bars, perf_plot["mean_cv_accuracy"]):
        ax.text(bar.get_x() + bar.get_width()/2, value + 0.004, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    
    fig.tight_layout()
    
    perf_png = os.path.join(outdir, f"{patient_id}_{model_name}_{atlas}_model_accuracy_{attn_scale}.png")
    
    fig.savefig(perf_png, dpi=300, bbox_inches="tight")
    
    plt.show()
    
    print(f"Saved plots to: {outdir}")



else:
    #coefficients across atlases and freqs
    coef_df = atlas_coefficient_results_df
  
    outdir = f"/media/christine/Samsung/attn_states_decoding/{patient_id}_{model_name}_coefs_bargraphs"
    os.makedirs(outdir, exist_ok=True)
    
    # -----------------------------
    # 1. Signed mean coefficients by frequency
    # -----------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    
    freq_order = ["theta", "alpha", "beta", "gamma", "HFA"]
    
    d = coef_df.set_index("frequency").reindex(freq_order)
    
    x = np.arange(len(freq_order))
    
    bars = ax.bar(x, d["mean_coefficient"])
    
    ax.axhline(0, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(freq_order)
    ax.set_ylabel(f"Mean {model_name} coefficient")
    ax.set_xlabel("Frequency band")
    ax.set_title(f"{patient_id}: Mean {model_name} coefficients by frequency ({attn_scale})")
    
    fig.tight_layout()
    plt.show()
    
    signed_png = os.path.join(outdir,f"{patient_id}_{model_name}_mean_coefficients_by_freq_{attn_scale}.png")
    fig.savefig(signed_png, dpi=300, bbox_inches="tight")