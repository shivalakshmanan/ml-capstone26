import os
import joblib
import numpy as np
import pandas as pd

# Load the models and supplementary artifacts
pipeline = joblib.load("ml-capstone26/models/delay_model.pkl")
threshold_df = pd.read_csv("ml-capstone26/models/delay_threshold.csv")
threshold = float(threshold_df["threshold"].iloc[0])

feature_order_df = pd.read_csv("ml-capstone26/models/feature_order.csv")
features_list = list(feature_order_df["feature"])

impute_df = pd.read_csv("ml-capstone26/models/impute_values.csv")
stored_medians = dict(zip(impute_df["column"], impute_df["median"]))

hub_segments_df = pd.read_csv("ml-capstone26/outputs/hub_segments.csv")

# Load the new shipments
new_raw_df = pd.read_csv("ml-capstone26/data/new_shipments.csv")
hubs_df = pd.read_csv("ml-capstone26/data/hubs.csv")

# Clean and build features using stored parameters
cleaned_new_df, _ = clean_shipments(new_raw_df, medians=stored_medians)
new_features_df = build_features(cleaned_new_df, hubs_df)

# Prepare inputs in the exact feature order
X_new = new_features_df[features_list]

# Predict probabilities
delay_prob = pipeline.predict_proba(X_new)[:, 1]
new_features_df["delay_probability"] = np.round(delay_prob, 4)
new_features_df["flagged"] = (new_features_df["delay_probability"] >= threshold).astype(int)

# Join hub segments
new_features_df = new_features_df.merge(hub_segments_df[["hub_id", "segment"]], on="hub_id", how="left")

# Apply action selection logic
def map_action(row):
    if row["flagged"] == 0:
        return "No action"
    elif row["segment"] == "Escalate":
        return "Escalate to hub manager"
    else:
        return "Proactive customer call"

new_features_df["action"] = new_features_df.apply(map_action, axis=1)

# Align to final format and export
final_output = new_raw_df[["shipment_id"]].merge(
    new_features_df[["shipment_id", "hub_id", "segment", "delay_probability", "flagged", "action"]],
    on="shipment_id",
    how="left"
)

final_output.to_csv("ml-capstone26/outputs/new_shipment_actions.csv", index=False)
# display(final_output.head(10))
