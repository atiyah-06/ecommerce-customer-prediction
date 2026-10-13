# ecommerce-customer-prediction
Machine learning pipeline to spot at-risk e-commerce customers before they leave.

## Key Business Insights
1. **Inactivity is the Top Warning Sign:** `inactivity_ratio` and `DaySinceLastOrder` are the strongest indicators of churn. When inactivity increases relative to tenure, churn risk rises sharply.
2. **New Customers are Sensitive:** Short-tenure customers drop off quickly if they experience complaints or long delivery distances.
3. **Action Plan:** Send retention offers automatically at 30–45 days of customer inactivity.

## Technical Approach
1. **Preprocessing & Feature Engineering:** Created custom metrics (`inactivity_ratio`, `order_frequency`), filled missing values using training set medians, and one-hot encoded categorical variables.
2. **Pipelines & Scaling:** Used Scikit-Learn `Pipeline` workflows with `StandardScaler` to prevent data leakage.
3. **Class Imbalance:** Applied `class_weight='balanced'` to optimize **Recall** so the model focuses on catching as many churners as possible.
4. **Models Trained:** Logistic Regression baseline vs. Decision Tree (`max_depth=7`).

## Feature Importance Methods
1. **Decision Tree:** Uses **Gini Impurity** to rank top structural split features (like `inactivity_ratio` at the root node).
2. **Logistic Regression:** Uses **Permutation Importance** on the test set to measure the exact drop in **Recall** when a feature column is scrambled.

## Model Limitations & Trade-offs
1. **Lower Precision (False Positives):** Balancing class weights causes the model to flag some non-churners as lost.
2. **Business Impact:** This trade-off is intentional, as the cost of losing a customer is higher than sending an unnecessary discount offer to an active user.

## Files in Repository
* `Ecommerce_Churn_Analysis.ipynb` — Executed Jupyter Notebook
* `decision_tree_flowchart.png` — Visualized Decision Tree diagram
* `feature_importances_comparison.png` — Side-by-side feature importance chart
