
import os
import pandas as pd
import numpy as np

os.makedirs("outputs", exist_ok=True)

# Read input data
raw_df = pd.read_csv("ml-capstone26/data/raw_shipments.csv")
hubs_df = pd.read_csv("ml-capstone26/data/hubs.csv")

# Extract "Before" metrics after removing duplicates but before any cleaning/filling
dup_free_raw = raw_df.drop_duplicates().copy()
missing_distance_before = dup_free_raw["distance_km"].isna().sum()

# missing_parcel_weight_before does not include sentinel rows (> 100)
weight_before_subset = dup_free_raw[~(dup_free_raw["parcel_weight_kg"] > 100)]
missing_parcel_weight_before = weight_before_subset["parcel_weight_kg"].isna().sum()

missing_declared_value_before = dup_free_raw["declared_value"].isna().sum()
missing_payment_mode_before = dup_free_raw["payment_mode"].isna().sum()

service_type_spellings_before = dup_free_raw["service_type"].nunique()

# Apply cleaning and feature engineering
clean_df, medians = clean_shipments(raw_df)
service_type_spellings_after = clean_df["service_type"].nunique()

modelling_df = build_features(clean_df, hubs_df)
modelling_df["is_delayed"] = (modelling_df["delivery_hours"] > modelling_df["promised_hours"]).astype(int)

# Extract "After" metrics on modelling_df
target_cols = FEATURES + ["delivery_hours", "is_delayed"]
nulls_in_features_and_target = modelling_df[target_cols].isna().sum().sum()

declared_value_mean_inr = modelling_df["declared_value_inr"].mean()
declared_value_median_inr = modelling_df["declared_value_inr"].median()
log_declared_value_mean = modelling_df["log_declared_value"].mean()
log_declared_value_median = modelling_df["log_declared_value"].median()

# Calculate outliers boundaries on delivery_hours using IQR method
q25 = modelling_df["delivery_hours"].quantile(0.25)
q75 = modelling_df["delivery_hours"].quantile(0.75)
iqr = q75 - q25
hours_lower_bound = q25 - 1.5 * iqr
hours_upper_bound = q75 + 1.5 * iqr

# Compile checks into DataFrame and round values
checks_data = {
    "Metric": [
        "missing_distance_before",
        "missing_parcel_weight_before",
        "missing_declared_value_before",
        "missing_payment_mode_before",
        "service_type_spellings_before",
        "service_type_spellings_after",
        "nulls_in_features_and_target",
        "declared_value_mean_inr",
        "declared_value_median_inr",
        "log_declared_value_mean",
        "log_declared_value_median",
        "hours_lower_bound",
        "hours_upper_bound"
    ],
    "value": [
        round(float(missing_distance_before), 3),
        round(float(missing_parcel_weight_before), 3),
        round(float(missing_declared_value_before), 3),
        round(float(missing_payment_mode_before), 3),
        round(float(service_type_spellings_before), 3),
        round(float(service_type_spellings_after), 3),
        round(float(nulls_in_features_and_target), 3),
        round(float(declared_value_mean_inr), 3),
        round(float(declared_value_median_inr), 3),
        round(float(log_declared_value_mean), 3),
        round(float(log_declared_value_median), 3),
        round(float(hours_lower_bound), 3),
        round(float(hours_upper_bound), 3)
    ]
}

part_a_checks = pd.DataFrame(checks_data)

# Save to file
modelling_df.to_csv("ml-capstone26/data/modelling_table.csv", index=False)
part_a_checks.to_csv("ml-capstone26/outputs/part_a_checks.csv", index=False)
# display(part_a_checks)
