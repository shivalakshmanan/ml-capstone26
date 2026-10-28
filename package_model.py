import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# Ensure output directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Load configuration and training data
config_df = pd.read_csv("ml-capstone26/outputs/part_c_final_config.csv")
chosen_threshold = float(config_df["threshold"].iloc[0])

df = pd.read_csv("ml-capstone26/data/modelling_table.csv")
X = df[FEATURES]
y = df["is_delayed"]

# Re-apply identical stratified split rules (70/15/15)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# Re-fit the exact pipeline on original training rows
model_params = {"random_state": 42}
final_pipeline = Pipeline([
    ("selector", SelectKBest(score_func=f_classif, k=7)),
    ("model", RandomForestClassifier(**model_params))
]).fit(X_train, y_train)

# Save serialized artifacts to models/
joblib.dump(final_pipeline, "ml-capstone26/models/delay_model.pkl")

pd.DataFrame({"threshold": [chosen_threshold]}).to_csv("ml-capstone26/models/delay_threshold.csv", index=False)

pd.DataFrame({"feature": FEATURES}).to_csv("ml-capstone26/models/feature_order.csv", index=False)

# Get medians calculated during clean_shipments stage
_ , medians_dict = clean_shipments(pd.read_csv("ml-capstone26/data/raw_shipments.csv"))
impute_df = pd.DataFrame(list(medians_dict.items()), columns=["column", "median"])
impute_df.to_csv("ml-capstone26/models/impute_values.csv", index=False)

# Reload verified model and perform consistency test
loaded_pipeline = joblib.load("ml-capstone26/models/delay_model.pkl")

# Predict probabilities for the test rows using both
probs_in_memory = final_pipeline.predict_proba(X_test)[:, 1]
probs_loaded = loaded_pipeline.predict_proba(X_test)[:, 1]

# Calculate verification metrics
test_rows_scored = float(len(X_test))
max_prob_diff = float(np.max(np.abs(probs_in_memory - probs_loaded)))

reload_check_df = pd.DataFrame({
    "metric": [
        "test_rows_scored",
        "reload_max_abs_probability_diff",
        "threshold"
    ],
    "value": [
        round(test_rows_scored, 3),
        round(max_prob_diff, 3),
        round(chosen_threshold, 3)
    ]
})

reload_check_df.to_csv("ml-capstone26/outputs/part_d_reload_check.csv", index=False)
# display(reload_check_df)

