#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 15:32:22 2026

@author: christine
"""

#reshape experience sampling responses

import pandas as pd
import os

pat = 'LH019'

log_dir = f'/media/christine/Samsung/exp_sampling/logs/data/{pat}/'

#log_file = 'lh019_inscapes01-04-28_15-03-25.csv'
log_file = 'lh019_betta02-04-28_14-52-21.csv'
#log_file = 'lh019_betta01-04-28_14-04-08.csv'
vid = "betta"

log_path = os.path.join(log_dir + log_file)

df = pd.read_csv(log_path)

response_wide_df = (
    df
    .drop_duplicates(subset=["pause_idx", "q_id"], keep="first")
    .pivot(index="pause_idx", columns="q_id", values="key")
    .reset_index()
)

response_wide_df.columns.name = None
response_wide_df = response_wide_df[
    ["pause_idx", "ext_int", "valence", "physio"]
]
response_wide_df["trial"] = response_wide_df["pause_idx"]
response_wide_df["pat"]=pat
response_wide_df["vid"]=vid
response_wide_df = response_wide_df[[
        "pat",
        "vid",
        "trial",
        "ext_int",
        "valence",
        "physio"
        ]
    ]

out_file = log_file.replace(".csv", "_wide_processed.csv")
out_path = os.path.join(log_dir + out_file)

response_wide_df.to_csv(out_path)