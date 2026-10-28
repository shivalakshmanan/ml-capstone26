
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

# read data
data = pd.read_csv("ml-capstone26/data/modelling_table.csv")
X = data[FEATURES]
y = data["delivery_hours"]

# Call train_test_split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Fit StandardScaler and transform the data sets
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Calculate metrics
train_rows = float(len(X_train))
val_rows = float(len(X_val))
test_rows = float(len(X_test))

train_scaled_max_abs_mean = float(np.max(np.abs(np.mean(X_train_scaled, axis=0))))
test_scaled_max_abs_mean = float(np.max(np.abs(np.mean(X_test_scaled, axis=0))))

# Write outputs/part_b_checks.csv
checks_df = pd.DataFrame({
    "metric": [
        "train_rows",
        "val_rows",
        "test_rows",
        "train_scaled_max_abs_mean",
        "test_scaled_max_abs_mean"
    ],
    "value": [
        round(train_rows, 3),
        round(val_rows, 3),
        round(test_rows, 3),
        round(train_scaled_max_abs_mean, 3),
        round(test_scaled_max_abs_mean, 3)
    ]
})

import os
os.makedirs("ml-capstone26/outputs", exist_ok=True)
checks_df.to_csv("ml-capstone26/outputs/part_b_checks.csv", index=False)
# display(checks_df)

# Predict the mean of y_train for every validation row and record its metrics.
y_train_mean = np.mean(y_train)
y_val_mean_pred = np.full(len(y_val), y_train_mean)
val_mean_mse = mean_squared_error(y_val, y_val_mean_pred)
val_mean_rmse = np.sqrt(val_mean_mse)
val_mean_r2 = r2_score(y_val, y_val_mean_pred)

# print("Validation Mean MSE:", val_mean_mse)
# print("Validation Mean RMSE:", val_mean_rmse)
# print("Validation Mean R2:", val_mean_r2)

depths = [1, 2, 3, 4, 6, 8, 12, None]
results = []

for d in depths:
    tree = DecisionTreeRegressor(max_depth=d, random_state=42)
    tree.fit(X_train_scaled, y_train)

    train_preds = tree.predict(X_train_scaled)
    val_preds = tree.predict(X_val_scaled)

    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))

    depth_str = "unlimited" if d is None else str(d)
    results.append({
        "max_depth": depth_str,
        "train_rmse": round(train_rmse, 3),
        "val_rmse": round(val_rmse, 3)
    })

tree_depth_df = pd.DataFrame(results)
tree_depth_df.to_csv("ml-capstone26/outputs/part_b_tree_depth.csv", index=False)
# display(tree_depth_df)

# optimal tree depth
otree_depth = int(tree_depth_df.loc[tree_depth_df['val_rmse'] == tree_depth_df['val_rmse'].min(), 'max_depth'].values[0])
# print(f"The calculated optimal tree depth is: {otree_depth}")

# Initialize lists to store metrics
models_metrics = []

# Helper to compute and return metric dictionary
def get_metrics(y_true, y_pred, model_name, split="val"):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {
        "model": model_name,
        "split": split,
        "MAE": round(mae, 3),
        "MSE": round(mse, 3),
        "RMSE": round(rmse, 3),
        "R2": round(r2, 3)
    }

# 1. Mean Baseline
baseline_metrics = get_metrics(y_val, y_val_mean_pred, "mean_baseline")
models_metrics.append(baseline_metrics)

# 2. linear Regression Pipeline
reg_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
]).fit(X_train, y_train)
reg_preds = reg_pipe.predict(X_val)
models_metrics.append(get_metrics(y_val, reg_preds, "linear_regression"))

# 3. Decision Tree (using optimal depth calculated)
dtree = DecisionTreeRegressor(max_depth=otree_depth, random_state=42).fit(X_train_scaled, y_train)
dtree_preds = dtree.predict(X_val_scaled)
models_metrics.append(get_metrics(y_val, dtree_preds, "decision_tree"))

# 4. Random Forest
rforest = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train_scaled, y_train)
rforest_preds = rforest.predict(X_val_scaled)
models_metrics.append(get_metrics(y_val, rforest_preds, "random_forest"))

# 5. Gradient Boosting
gboost = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42).fit(X_train_scaled, y_train)
gboost_preds = gboost.predict(X_val_scaled)
models_metrics.append(get_metrics(y_val, gboost_preds, "gradient_boosting"))

# Compile into DataFrame & save
metrics_df = pd.DataFrame(models_metrics)
metrics_df.to_csv("ml-capstone26/outputs/part_b_metrics.csv", index=False)
# display(metrics_df)

learn_rate = [0.01, 0.1, 1.0]
experiment_results = []

# Learning-rate experiment
for l in learn_rate:
    gboost = GradientBoostingRegressor(n_estimators=100, learning_rate=l, random_state=42).fit(X_train_scaled, y_train)
    preds = gboost.predict(X_val_scaled)

    rmse = np.sqrt(mean_squared_error(y_val, preds))
    r2 = r2_score(y_val, preds)

    experiment_results.append({
        "learning_rate": l,
        "val_rmse": round(rmse, 3),
        "val_r2": round(r2, 3)
    })

experiment_df = pd.DataFrame(experiment_results)
experiment_df.to_csv("ml-capstone26/outputs/part_b_gb_learning_rate.csv", index=False)
# display(experiment_df)

# Combine training and validation data (840 + 180 = 1,020 rows)
X_train_val_raw = pd.concat([X_train, X_val], axis=0)
X_train_val_scaled = np.concatenate([X_train_scaled, X_val_scaled], axis=0)
y_train_val = pd.concat([y_train, y_val], axis=0)

# Define the shared KFold cross-validator
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# two best models based on validation RMSE are:
# Best 1: Linear Regression (RMSE: 4.891)
# Best 2: Gradient Boosting (RMSE: 4.916)

# cross-validation for Linear Regression Pipeline
lr_cv_scores = cross_val_score(reg_pipe, X_train_val_raw, y_train_val, cv=kf, scoring="r2")

# Run cross-validation for Gradient Boosting
gb_cv_scores = cross_val_score(gboost, X_train_val_scaled, y_train_val, cv=kf, scoring="r2")

# Write outputs/part_b_cv.csv
cv_results = pd.DataFrame([
    {
        "model": "linear_regression",
        "mean_r2": round(np.mean(lr_cv_scores), 3),
        "std_r2": round(np.std(lr_cv_scores), 3)
    },
    {
        "model": "gradient_boosting",
        "mean_r2": round(np.mean(gb_cv_scores), 3),
        "std_r2": round(np.std(gb_cv_scores), 3)
    }
])
cv_results.to_csv("ml-capstone26/outputs/part_b_cv.csv", index=False)
# print("Cross-Validation Results:")
# display(cv_results)

# 2. Score the final best model (linear_regression) on the test rows
# It must remain fitted only on the original training rows (reg_pipe is already fitted on X_train)
test_preds = reg_pipe.predict(X_test)

test_mae = mean_absolute_error(y_test, test_preds)
test_mse = mean_squared_error(y_test, test_preds)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test, test_preds)

# Read existing metrics
metrics_df = pd.read_csv("ml-capstone26/outputs/part_b_metrics.csv")

# Create the new test row
test_row = pd.DataFrame([{
    "model": "linear_regression",
    "split": "test",
    "MAE": round(test_mae, 3),
    "MSE": round(test_mse, 3),
    "RMSE": round(test_rmse, 3),
    "R2": round(test_r2, 3)
}])

# Append and save the updated metrics table
metrics_df = pd.concat([metrics_df, test_row], ignore_index=True)
metrics_df.to_csv("ml-capstone26/outputs/part_b_metrics.csv", index=False)
# print("\nUpdated Metrics Table (including final test score):")
# display(metrics_df)
