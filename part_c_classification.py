import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Ensure output directory exists
os.makedirs("ml-capstone26/outputs", exist_ok=True)

# Define the features exactly in the same pinned order
FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]

# Read input dataset
df = pd.read_csv("ml-capstone26/data/modelling_table.csv")
X = df[FEATURES]
y = df["is_delayed"]

# Create stratified train/validation/test splits (70/15/15) matching split rules
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# Helper function to compute all required stats
def get_class_metrics(model_name, y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = accuracy_score(y_true, y_pred)
    # Write precision as 0 if requested, or if no positive predictions
    if model_name == "blind_all_on_time":
        prec = 0.0
    else:
        prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {
        "model": model_name,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "accuracy": round(float(acc), 3),
        "precision": round(float(prec), 3),
        "recall": round(float(rec), 3),
        "f1": round(float(f1), 3)
    }

# 1. Blind Baseline (all on time, predicts 0)
y_val_blind = np.zeros(len(y_val))
blind_metrics = get_class_metrics("blind_all_on_time", y_val, y_val_blind)

# 2. Logistic Regression Pipeline
lr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(random_state=42, max_iter=1000))
]).fit(X_train, y_train)
lr_preds = lr_pipeline.predict(X_val)
lr_metrics = get_class_metrics("logistic_regression", y_val, lr_preds)

# 3. Decision Tree: Hyperparameter Tuning for optimal depth d
depths_to_try = [2, 3, 4, 5, 6, 8, 10]
best_d = None
best_f1 = -1.0
best_dt_metrics = None

for d in depths_to_try:
    dt_clf = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42).fit(X_train, y_train)
    dt_preds = dt_clf.predict(X_val)
    metrics = get_class_metrics(f"decision_tree_depth_{d}", y_val, dt_preds)

    # Select highest validation F1, ties -> smaller d
    if metrics["f1"] > best_f1:
        best_f1 = metrics["f1"]
        best_d = d
        best_dt_metrics = metrics
    elif metrics["f1"] == best_f1:
        if best_d is None or d < best_d:
            best_d = d
            best_dt_metrics = metrics

# 4. Random Forest Default Classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
rf_preds = rf_clf.predict(X_val)
rf_metrics = get_class_metrics("random_forest_default", y_val, rf_preds)

# Compile all rows into final DataFrame and save
part_c_models = pd.DataFrame([
    blind_metrics,
    lr_metrics,
    best_dt_metrics,
    rf_metrics
])
part_c_models.to_csv("ml-capstone26/outputs/part_c_models.csv", index=False)
# print("Saved outputs/part_c_models.csv")

# Feature Importances with Random Forest
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
rf_importances = rf_clf.feature_importances_

# F-scores with SelectKBest
selector = SelectKBest(score_func=f_classif, k="all")
selector.fit(X_train, y_train)
f_scores = selector.scores_

# Compile feature scores DataFrame
feature_scores_df = pd.DataFrame({
    "feature": FEATURES,
    "f_score": f_scores,
    "rf_importance": rf_importances
})

# Add rankings and noise column flags
feature_scores_df["f_rank"] = feature_scores_df["f_score"].rank(ascending=False, method="min").astype(int)
feature_scores_df["is_noise"] = feature_scores_df["feature"].isin(NOISE_COLUMNS)

# Reorder columns as requested
feature_scores_df = feature_scores_df[["feature", "f_score", "f_rank", "rf_importance", "is_noise"]]
feature_scores_df.to_csv("ml-capstone26/outputs/part_c_feature_scores.csv", index=False)
# display(feature_scores_df)

# validate sum of rf_importance when is_noise = True
# print(feature_scores_df[feature_scores_df["is_noise"]]["rf_importance"].sum())

# Feature Selection Loop
k_values = [5, 7, 10, 12, 17]
k_results = []

for k in k_values:
    pipe = Pipeline([
        ("select", SelectKBest(score_func=f_classif, k=k)),
        ("rf", RandomForestClassifier(n_estimators=100, random_state=42))
    ]).fit(X_train, y_train)
    val_preds = pipe.predict(X_val)
    f1 = f1_score(y_val, val_preds, zero_division=0)
    k_results.append({
        "k": k,
        "val_f1": round(float(f1), 3)
    })

k_df = pd.DataFrame(k_results)
k_df.to_csv("ml-capstone26/outputs/part_c_selectk.csv", index=False)
# display(k_df)

# Feature Selection (k=7)
selector_k7 = SelectKBest(score_func=f_classif, k=7)
X_train_k7 = selector_k7.fit_transform(X_train, y_train)
X_val_k7 = selector_k7.transform(X_val)

# Hyperparameter Grid Setup
param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [4, 6, 8],
    "min_samples_leaf": [1, 5, 10]
}

# GridSearchCV Setup
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    n_jobs=-1
).fit(X_train_k7, y_train)
grid_val_preds = grid_search.predict(X_val_k7)
grid_f1 = f1_score(y_val, grid_val_preds, zero_division=0)

# RandomizedSearchCV Setup
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_grid,
    n_iter=9,
    cv=5,
    random_state=42,
    n_jobs=-1
).fit(X_train_k7, y_train)
random_val_preds = random_search.predict(X_val_k7)
random_f1 = f1_score(y_val, random_val_preds, zero_division=0)

# Default Forest (Untuned) on the same 7 features
default_forest = RandomForestClassifier(random_state=42).fit(X_train_k7, y_train)
default_val_preds = default_forest.predict(X_val_k7)
default_f1 = f1_score(y_val, default_val_preds, zero_division=0)

# 6. Build the comparison table as specified
tuning_results = [
    {
        "search": "grid_search",
        "candidates": 27,
        "cv_folds": 5,
        "total_fits": 27 * 5,
        "best_params": str(grid_search.best_params_),
        "best_cv_accuracy": round(grid_search.best_score_, 3),
        "val_f1": round(grid_f1, 3)
    },
    {
        "search": "randomized_search",
        "candidates": 9,
        "cv_folds": 5,
        "total_fits": 9 * 5,
        "best_params": str(random_search.best_params_),
        "best_cv_accuracy": round(random_search.best_score_, 3),
        "val_f1": round(random_f1, 3)
    },
    {
        "search": "default_forest_same_7_features",
        "candidates": 1,
        "cv_folds": 0,
        "total_fits": 1,
        "best_params": "defaults (n_estimators=100, max_depth=None, min_samples_leaf=1)",
        "best_cv_accuracy": np.nan,
        "val_f1": round(default_f1, 3)
    }
]

tuning_df = pd.DataFrame(tuning_results)
os.makedirs("ml-capstone26/outputs", exist_ok=True)
tuning_df.to_csv("ml-capstone26/outputs/part_c_tuning.csv", index=False)
# display(tuning_df)

# Load tuning results
# tuning_df = pd.read_csv("ml-capstone26/outputs/part_c_tuning.csv")

# Define the custom sorting / tie-breaking order
# Tie-breaker priority: default_forest_same_7_features (0), grid_search (1), randomized_search (2)
priority_map = {
    "default_forest_same_7_features": 0,
    "grid_search": 1,
    "randomized_search": 2
}
tuning_df["priority"] = tuning_df["search"].map(priority_map)

# Sort by val_f1 descending, then by priority ascending
best_row = tuning_df.sort_values(by=["val_f1", "priority"], ascending=[False, True]).iloc[0]

chosen_search = best_row["search"]
chosen_val_f1 = best_row["val_f1"]
best_params_str = best_row["best_params"]

# print(f"Selected configuration source: {chosen_search} (Validation F1: {chosen_val_f1})")

# Parse hyperparameters
if chosen_search == "default_forest_same_7_features":
    # Default parameters
    model_params = {"random_state": 42}
else:
    # Parse dictionary from string representation
    model_params = ast.literal_eval(best_params_str)
    model_params["random_state"] = 42

# Build the final Pipeline & fit on X_train
final_pipeline = Pipeline([
    ("selector", SelectKBest(score_func=f_classif, k=7)),
    ("model", RandomForestClassifier(**model_params))
]).fit(X_train, y_train)
# print(final_pipeline)

# Predicted probabilities on the validation set using final_pipeline
val_probs = final_pipeline.predict_proba(X_val)[:, 1]

# Build exact thresholds
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
threshold_results = []

for t in thresholds:
    preds = (val_probs >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()
    acc = accuracy_score(y_val, preds)
    prec = precision_score(y_val, preds, zero_division=0)
    rec = recall_score(y_val, preds, zero_division=0)
    f1 = f1_score(y_val, preds, zero_division=0)

    threshold_results.append({
        "threshold": t,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "accuracy": round(float(acc), 3),
        "precision": round(float(prec), 3),
        "recall": round(float(rec), 3),
        "f1": round(float(f1), 3)
    })

thresholds_df = pd.DataFrame(threshold_results)
os.makedirs("ml-capstone26/outputs", exist_ok=True)
thresholds_df.to_csv("ml-capstone26/outputs/part_c_thresholds.csv", index=False)

# Apply operational rule to find the alert threshold
# Highest t whose validation recall is >= 0.80. If none, use 0.1.
eligible_thresholds = thresholds_df[thresholds_df["recall"] >= 0.80]
if not eligible_thresholds.empty:
    alert_threshold = float(eligible_thresholds["threshold"].max())
else:
    alert_threshold = 0.1

# print(f"Selected Alert Threshold: {alert_threshold}")
# display(thresholds_df)

# Ensure alert_threshold is defined (fallback to 0.5 as computed on validation if missing)
try:
    chosen_threshold = alert_threshold
except NameError:
    chosen_threshold = 0.5

# Get predicted probabilities on the test set
test_probs = final_pipeline.predict_proba(X_test)[:, 1]

# Score using the chosen alert threshold
test_preds = (test_probs >= chosen_threshold).astype(int)

# Compute test metrics
tn, fp, fn, tp = confusion_matrix(y_test, test_preds).ravel()
accuracy = accuracy_score(y_test, test_preds)
precision = precision_score(y_test, test_preds, zero_division=0)
recall = recall_score(y_test, test_preds, zero_division=0)
f1 = f1_score(y_test, test_preds, zero_division=0)

# Write outputs/part_c_final_test.csv
final_test_df = pd.DataFrame([{
    "threshold": chosen_threshold,
    "TN": int(tn),
    "FP": int(fp),
    "FN": int(fn),
    "TP": int(tp),
    "accuracy": round(float(accuracy), 3),
    "precision": round(float(precision), 3),
    "recall": round(float(recall), 3),
    "f1": round(float(f1), 3)
}])

os.makedirs("outputs", exist_ok=True)
final_test_df.to_csv("ml-capstone26/outputs/part_c_final_test.csv", index=False)
print("Final Test Performance at Threshold:", chosen_threshold)
# display(final_test_df)

# Extract hyperparameters for config table
config_df = pd.DataFrame([{
    "selector_k": 7,
    "n_estimators": 100,
    "max_depth": "unlimited",
    "min_samples_leaf": 1,
    "threshold": chosen_threshold,
    "chosen_from": "default_forest_same_7_features"
}])

config_df.to_csv("ml-capstone26/outputs/part_c_final_config.csv", index=False)
# print("\nFinal Configuration Saved:")
# display(config_df)
