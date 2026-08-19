#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 22:49:53 2026

@author: christine
"""
import pandas as pd

patients = ['NS217','NS219','NS221','LH019']

atlas = "aparc_aseg"

aparc_aseg_networks = ['ctx-rh-caudalanteriorcingulate', 'ctx-lh-caudalanteriorcingulate',
       'Left-Cerebral-White-Matter', 'Right-Cerebral-White-Matter',
       'CC_Central','ctx-lh-rostralanteriorcingulate',
      'Left-Caudate', 'ctx-rh-superiorfrontal',
       'ctx-lh-superiorfrontal', 'ctx-lh-rostralmiddlefrontal',
       'ctx-lh-caudalmiddlefrontal']

yeo7_networks = ["Visual", "Somatomotor", "Dorsal Attention", "Ventral Attention", "Limbic", "Frontoparietal", "Default"]

yeo17_networks = [
    "Visual Central (Visual A)",
    "Visual Peripheral (Visual B)",
    "Somatomotor A",
    "Somatomotor B",
    "Dorsal Attention A",
    "Dorsal Attention B",
    "Salience / Ventral Attention A",
    "Salience / Ventral Attention B",
    "Limbic A",
    "Limbic B",
    "Control C",
    "Control A",
    "Control B",
    "Temporal Parietal",
    "Default C",
    "Default A",
    "Default B"
]


all_patient_atlas_dfs = []

for patient_id in patients:

    input_file = f"/media/christine/Samsung/Movie_data/final_power_dataframes/{patient_id}_final_power_dataframe.csv"
    df = pd.read_csv(input_file)
        
    if atlas == "Yeo7":
        atlas_networks = yeo7_networks
        df_atlas = df[(df["exclude"] == 0) & (df[atlas].isin(yeo7_networks))].copy()
        
    if atlas == "aparc_aseg":
        atlas_networks =aparc_aseg_networks
        df_atlas = df[(df["exclude"] == 0) & (df[atlas].isin(aparc_aseg_networks))].copy()
    
    elif atlas == "Yeo17":
       df_atlas = df[(df["exclude"] == 0) & (df[atlas].isin(yeo17_networks))].copy()
    elif atlas == "Desikan_Killiany":
        df_atlas = df[
            (df["exclude"] == 0) &
            (~df[atlas].isin(["unknown", "Out"]))
            & (~df[atlas].astype(str).str.contains(r"\d", regex=True))
        ].copy()

    df_atlas = (
        df_atlas
        .groupby(["subject", "recording", "trial", "time", "freq", atlas], dropna=False)
        .agg(
            power=("power", "mean"),
            n_contacts=("channel", "nunique"),
            attn_label=("attn_label", "first"),
            valence=("valence", "first"),
            physio=("physio", "first")
        )
        .reset_index()
    )

    df_atlas["atlas_freq"] = df_atlas[atlas].astype(str) + "_" + df_atlas["freq"].astype(str)

    all_patient_atlas_dfs.append(df_atlas)

combined_atlas_df = pd.concat(all_patient_atlas_dfs, ignore_index=True)

output_file = f"/media/christine/Samsung/Movie_data/final_power_dataframes/combined_{atlas}_power_dataframe.csv"

combined_atlas_df.to_csv(output_file, index=False)

print(combined_atlas_df.shape)
print(combined_atlas_df["subject"].value_counts())
print(combined_atlas_df[atlas].value_counts())
print(f"Saved: {output_file}")