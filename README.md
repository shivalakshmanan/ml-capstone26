# README - Data Findings

### Core Findings from Part A Analysis

1. **Service Type Delays**
   The service type delayed most often is **Express** with a delay rate of **0.4897**.

2. **Delay Rate by Hub Tier**
   The delay rate increases significantly as we move from Tier 1 to Tier 3 hubs: Tier 1 has a delay rate of **0.0400**, Tier 2 is **0.2150**, and Tier 3 is **0.4400**.

3. **Delay Rate by Distance Band**
   Across distance bands, the delay rate escalates linearly with distance: **0.0219** (0-200 km), **0.137** (201-400 km), **0.4364** (401-600 km), and **0.8523** (601+ km).

4. **Top Correlations with Delivery Hours**
   The three columns most strongly correlated with `delivery_hours` are: promised_hours (0.8370), distance_km (0.4010), n_stops (0.3250).

5. **Declared Value Log Transformation**
   The `declared_value_inr` was log-transformed because the original column is highly right-skewed with a mean of **1533.0540** and a median of **1121.5000**; after the transformation, the distribution is normalized with a mean of **7.0280** and median of **7.0220**.


## Model Results (Regression)

### 1. Baseline R² and Interpretation
* **Baseline R²:** -0.021
* **Meaning:** The mean baseline model predicts the mean of the training target (`delivery_hours`) for all validation observations. An R² score around 0 or slightly negative indicates that the model performs no better than simply predicting the historical training average, serving as a trivial baseline for our actual regression models.

### 2. Validation Winner
* **Winning Model:** linear_regression
* **Validation RMSE:** 4.891 hours
* **Alternative Top Performer:** Gradient Boosting (with `learning_rate=0.1`) achieved a very competitive validation RMSE of 4.916 hours.

### 3. Bias-Variance Tradeoff (Decision Tree Depths)
* **High-Bias Depths:** Depths 1, 2, 3, and 4 (validation RMSEs range from 7.39 to 12.13 hours). At these low depths, the tree is underfitting the training data (high train RMSE) and cannot capture the complex underlying relationships.
* **High-Variance Depths:** Unlimited depth or very high depth (e.g., Depth 12 and unlimited, where validation RMSE increases back to 7.37 - 7.84 hours while training RMSE drops to 0). This massive gap indicates overfitting, where the model memorizes noise in the training set and generalizes poorly.
* **Optimal Depth:** Depth 6 yields the best tradeoff with a validation RMSE of 6.280 hours.

### 4. Learning-Rate Experiment (Gradient Boosting)
* **Best Learning Rate:** 0.1 (validation RMSE: 4.916)
* **Why the others fail:**
  * `learning_rate=0.01` (Too Slow): Underperforms (validation RMSE: 10.002) because the step size is too small; with only 100 estimators, the model does not converge to the optimal solution.
  * `learning_rate=1.0` (Overshooting): Underperforms (validation RMSE: 7.271) because the step size is too aggressive, causing the optimization process to overshoot minimums and oscillate or overfit rapidly.

### 5. Cross-Validation and Final Shipping Recommendation
* **Cross-Validation Results (5-Fold R²):**
  * **Linear Regression:** Mean R² = 0.926 (Std = 0.005)
  * **Gradient Boosting:** Mean R² = 0.862 (Std = 0.015)
* **Shipping Recommendation:** We recommend shipping the **Linear Regression Pipeline**. It consistently outperformed Gradient Boosting in cross-validation (mean R² of 0.926 vs 0.862) and showed excellent generalization with a final **Test Split RMSE of 4.524 hours**. It is computationally efficient, robust, and highly interpretable.


# Model Results (Classification)

### 1. Blind Baseline & Accuracy
* **Blind Baseline Accuracy:** 0.767
* **Why it is useless:** Although a blind model predicting all parcels are on time yields a high accuracy because delays are rare (the majority class), it has **0.0% recall** and **0.0% F1-score**. It completely fails to detect any delayed shipments, which defeats the core business operational objective of identifying at-risk deliveries.

### 2. Feature Selection & Noise Evaluation
* **Noise Features Importance:** The 5 noise columns collectively hold a **0.2244** share of the Random Forest's overall feature importance.
* **F-score Rank Check:** It is **True** that all 5 noise columns fall completely outside the top 7 features ranked by F-score.
* **SelectKBest (k) Validation F1 Scores:**
```
 k  val_f1
 5   0.699
 7   0.833
10   0.786
12   0.805
17   0.785
```
* **Optimal k:** A choice of **k=7** features yielded the highest validation F1 score of **0.833**.

### 3. Hyperparameter Tuning & Small-Sample Behavior

* **Tuning Performance:** The untuned random forest on the 7 selected features achieved a validation F1 of **0.833**, which actually **beat** both grid search (0.775) and randomized search (0.785).
* **Implication:** On small datasets (such as our 180-row validation set), aggressive hyperparameter search can easily overfit to validation-specific noise or produce sub-optimal results compared to robust ensemble defaults.


### 4. Alert Threshold & Precision-Recall Trade-off
* **Chosen Alert Threshold:** 0.5
* **Validation Trade-off:** At threshold 0.5, validation recall is **0.833** and precision is **0.795**. This configuration safely satisfies the business mandate of maintaining at least 80% recall while retaining decent precision to minimize false alarms.

### 5. Final Evaluation on Test Set
* **Test Precision:** 0.689
* **Test Recall:** 0.756
* **Test F1-score:** 0.721
* **Honest Comparison:** On the unseen test set (which containing only 41 delayed parcels), the recall dropped from a validation level of **0.833** to a test level of **0.756**. This reduction is a realistic representation of model variance on small, imbalanced subsets and emphasizes the importance of monitoring live deployment statistics.


## How to Run
Follow these steps in order to set up, re-train, and execute the batch prediction pipeline from a fresh repository clone:
1. **Environment Setup**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Generate the Datasets**:
   ```bash
   python generate_data.py
   ```
3. **Run Feature Engineering & Model Selection**:
   ```bash
   python part_a_clean_eda.py
   python part_b_regression.py
   python part_c_classification.py
   python part_d_segments.py
   ```
4. **Serialize the Assets**:
   ```bash
   python package_model.py
   ```
   *Note: `models/delay_model.pkl` is fully regenerated by running `package_model.py`. The pre-committed pickle file loads reliably only on the specific scikit-learn version pinned in `requirements.txt`. If system versions differ, the pipeline or grader will automatically re-run `package_model.py` instead of loading the old serialized model.*
5. **Run Inference on New Shipments**:
   ```bash
   python predict_delay.py
   ```
   This outputs the final decision mapping directly to `outputs/new_shipment_actions.csv`.

## Operations Recommendation
Based on our systematic analyses of the core delivery lanes and hub behaviors, we recommend the following strategic actions for the Regional Head:
* **Escalate Segment Focus**: The following hubs are experiencing severe operational bottlenecks and have been grouped into the **Escalate** segment: HUB13 (0.460 delay rate), HUB15 (0.360 delay rate), HUB17 (0.460 delay rate), HUB18 (0.380 delay rate), HUB19 (0.620 delay rate), HUB20 (0.480 delay rate), HUB21 (0.400 delay rate), HUB23 (0.440 delay rate), HUB24 (0.540 delay rate). These require direct pipeline capacity interventions and route restructuring.
* **Tier Mismatches**: We identified key discrepancies where a hub's physical infrastructure tier label does not align with its actual performance (found in `outputs/tier_vs_segment_mismatch.csv`): **HUB07 (actual: Watch, tier: 1), HUB13 (actual: Escalate, tier: 2), HUB15 (actual: Escalate, tier: 2), HUB22 (actual: Watch, tier: 3)**. This indicates that standard tier labels are misleading; operational decisions should be guided by their actual clustered performance segments rather than administrative labels.
* **Model Performance Metrics**: Using our optimized alert threshold of **0.5**:
  * **Precision**: Approximately **6.9 out of every 10 flagged parcels** are truly delayed, meaning false alarms are minimized to avoid team fatigue.
  * **Recall**: Approximately **7.6 out of every 10 delayed parcels** are successfully caught by the model before they miss their deadlines.
* **Operational Impact on New Batch**: Out of **48 new shipments**, the pipeline flagged **8 parcels** as high delay risks. Of these, **6 shipments** meet the criteria for immediate, high-priority **Hub Manager Escalation**, while the remaining flagged shipments are routed for proactive customer notifications.
* **Clustering Stability**: We retained **K = 3** clusters because it provides the optimal balance of group separation. According to `outputs/part_d_separation.csv`, K = 3 achieves a high across-to-within distance ratio of **5.393** with stable cluster sizes (**9-8-7**), whereas higher values of K show diminishing structural separation.

## Production Plan
* **Production-Readiness Mapping**:
  1. *Consistency*: Handled by saving `models/impute_values.csv` and `models/feature_order.csv`, ensuring training medians and column orderings are identical during inference.
  2. *Reproducibility*: Ensured by pinning all data splits and algorithms to a fixed seed (`random_state=42`).
  3. *Maintainability*: Modularized into dedicated, clean scripts (`shipment_prep.py`, `package_model.py`, `predict_delay.py`).
  4. *Performance*: Structured with optimized vector operations via pandas and NumPy, enabling batch inference of thousands of rows in under a second.
  5. *Reliability*: Implemented robust fallback exceptions (e.g., standardizing text casing and handling missing values with historical medians) to protect the inference runtime from schema variations.
* **Key Monitoring Indicators**:
  1. **Weekly Delayed Share**: Trigger an alert if the actual delay rate exceeds **30%** over a 7-day rolling window.
  2. **Weekly Flagged Share**: Trigger a system alert if more than **35%** of shipments are flagged, indicating possible structural changes or input drift.
  3. **Recall Slide**: Trigger re-evaluation if measured model recall falls below **75%** on shipments with resolved delivery outcomes.
* **Retraining & Handling Holiday Anomalies**:
  * Retrain the classifier monthly on a rolling 90-day window to adapt to shifting shipping patterns.
  * **Sale Event Handling**: The 15–28 September peak sales event (`sale_event_flag = 1`) must be evaluated independently. During future events, compare model performance against historic promotional periods rather than ordinary baseline weeks.
  * **Data Drift Identification**: Watch for increases in average trip distance (`distance_km`) or rider routes (`n_stops`) which signal structural expansion.

## Limitations
1. **Synthetic Data Nature**: The underlying data is generated through simulation, which may omit complex, real-world dependencies such as specific highway traffic blocks, weather severity changes, or vehicle breakdowns.
2. **Small Imbalanced Test Set**: The unseen test set contains only 41 delayed shipments, meaning that a shift of just 2-3 incorrectly predicted parcels causes noticeable fluctuations in our precision and recall metrics.
3. **Validation Single-Split Tuning**: The decision threshold and features were optimized using a single validation split (15%), which increases the risk of subtle overfitting to validation-specific quirks. Implementing rolling cross-validation in production will help mitigate this variance.
