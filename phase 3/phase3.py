# phase3.py

# ----------------------------
# Phase 3: Model Iteration & Final Evaluation
# ----------------------------
# Author: (Your Name)
# Date: (Today's Date)
#
# This script:
#   1. Loads the cleaned housing insecurity dataset (from Phase 2).
#   2. Repeats necessary preprocessing (Yes/No → 1/0, dummy encoding).
#   3. Performs stepwise logistic regression to identify key features.
#   4. Splits data into train/test and scales numeric features.
#   5. Compares three models (Logistic Regression, Random Forest, XGBoost)
#      using 5-fold cross-validation (AUC).
#   6. Fits a final logistic model and evaluates on the test set
#      (confusion matrix, sensitivity, specificity, AUC).
#   7. Computes VIF for the final logistic model.
# ----------------------------

import pandas as pd
import numpy as np

# For logistic regression by statsmodels
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.sm_exceptions import HessianInversionWarning

# For sklearn utilities
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)

import numpy as np
import statsmodels.api as sm

import warnings
warnings.filterwarnings("ignore", category=HessianInversionWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------
# 1. Load cleaned dataset
# ----------------------------
# Assumes 'housing_insecurity_prediction.xlsx' has a sheet named 'cleaned'
df = pd.read_excel('housing_insecurity_prediction.xlsx', sheet_name='cleaned')

# ----------------------------
# 2. Preprocessing (repeat key steps from Phase 2)
# ----------------------------

# 2.1 Drop any leftover PII/irrelevant columns if present
cols_to_drop = [
    'Start Date', 'End Date', 'Response Type', 'IP Address', 'Progress',
    'Duration (in seconds)', 'Finished', 'Recorded Date', 'Response ID',
    'Recipient Last Name', 'Recipient First Name', 'Location Latitude',
    'Location Longitude', 'Distribution Channel', 'User Language',
    'E-mail (this e-mail will be used to distribute your giftcard)',
    'Please provide the name of the community college you attended.',
    # The long follow-up interview question from Phase 2
    'We would like to follow-up with you on your responses with an interview. The interview will take a maximum of 30-45 minutes. As a result of COVID-19, we will conduct interviews via a teleconference tool.  For your participation in the interview, you will receive a $25 gift-card to Wal-Mart. Are you willing to participate in a follow-up interview?'
]
for col in cols_to_drop:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)

# 2.2 Split multi-choice columns into dummy variables
def split_multichoice_columns(df, column_name):
    if column_name in df.columns:
        dummies = df[column_name].str.get_dummies(sep=', ')
        df = pd.concat([df.drop(columns=[column_name]), dummies], axis=1)
    return df

df = split_multichoice_columns(
    df,
    'How do you usually describe your race and/or ethnicity?'
)
df = split_multichoice_columns(
    df,
    'Which of the following ways do you pay for the expenses associated with attending college? (check all that apply)'
)

# 2.3 Convert Yes/No columns to binary (1/0)
yes_no_cols = [
    'In the past 12 months, did you couch surf – that is, moved from one temporary housing arrangement to another because you had no other place to live?',
    'Have you ever been in foster care?',
    # If there are additional Yes/No columns used later, add here
]
for col in yes_no_cols:
    if col in df.columns:
        df[col] = df[col].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)

# 2.4 Define target (y) and predictor matrix (X)
target_col = (
    'In the past 12 months, did you couch surf – that is, moved from one temporary housing arrangement to another because you had no other place to live?'
)
y = df[target_col].astype(int)
X = df.drop(columns=[target_col])

# 2.5 Ensure all remaining columns are numeric (dummy or numeric)
# Try to convert any object columns to numeric if possible; otherwise drop non-numerics
for col in X.columns:
    if X[col].dtype == 'object':
        # Attempt numeric conversion; if fails, drop column
        try:
            X[col] = pd.to_numeric(X[col])
        except:
            X.drop(columns=[col], inplace=True)

# ----------------------------
# 3. Stepwise feature selection (re-derive Phase 2 predictors)
# ----------------------------
def stepwise_selection(
    X,
    y,
    threshold_in=0.05,
    threshold_out=0.10,
    verbose=False
):
    """
    Perform a forward–backward stepwise selection based on p-values from statsmodels.Logit.
    Returns the list of selected features.
    """
    included = []
    while True:
        changed = False
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded, dtype=float)

        # Forward step: consider adding each excluded feature
        for new_col in excluded:
            try:
                X_temp = sm.add_constant(X[included + [new_col]])
                model = sm.Logit(y, X_temp).fit(disp=0, maxiter=100)
                new_pval[new_col] = model.pvalues[new_col]
            except:
                new_pval[new_col] = np.nan

        # Add the feature with lowest p-value below threshold_in
        if not new_pval.empty:
            best_pval = new_pval.min()
            if best_pval < threshold_in:
                best_feature = new_pval.idxmin()
                included.append(best_feature)
                changed = True
                if verbose:
                    print(f'Add {best_feature:30} with p-value {best_pval:.4f}')

        # Backward step: consider removing based on p-values
        if included:
            X_temp = sm.add_constant(X[included])
            model = sm.Logit(y, X_temp).fit(disp=0, maxiter=100)
            # Exclude intercept (first p-value)
            pvalues = model.pvalues.iloc[1:]
            worst_pval = pvalues.max()
            if worst_pval > threshold_out:
                worst_feature = pvalues.idxmax()
                included.remove(worst_feature)
                changed = True
                if verbose:
                    print(f'Remove {worst_feature:30} with p-value {worst_pval:.4f}')

        if not changed:
            break

    return included

# Split data for feature selection (using 70% of data)
X_train_fs, X_unused, y_train_fs, y_unused = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

selected_feats = stepwise_selection(X_train_fs, y_train_fs, verbose=True)
print("\nSelected features by stepwise selection:")
print(selected_feats)

# ----------------------------
# 4. Split data into train/test & scale numeric features
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X[selected_feats], y, test_size=0.3, stratify=y, random_state=42
)

# Identify numeric columns for scaling (all of them should already be numeric/dummy)
numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train[numeric_cols]),
    columns=numeric_cols,
    index=X_train.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test[numeric_cols]),
    columns=numeric_cols,
    index=X_test.index
)

# ----------------------------
# 5. Model refinement & comparison
# ----------------------------
# We'll use 5-fold stratified CV and evaluate AUC for each classifier.

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 5.1 Logistic Regression (sklearn) with l2 penalty
logreg = LogisticRegression(
    penalty='l2', solver='lbfgs', max_iter=1000, random_state=42
)
scores_logreg = cross_val_score(
    logreg,
    X_train_scaled,
    y_train,
    cv=cv,
    scoring='roc_auc'
)
print(f"\nLogistic Regression 5-fold CV AUC: {scores_logreg.mean():.3f} ± {scores_logreg.std():.3f}")

# 5.2 Random Forest
rf = RandomForestClassifier(
    n_estimators=200, max_depth=5, random_state=42
)
scores_rf = cross_val_score(
    rf,
    X_train_scaled,
    y_train,
    cv=cv,
    scoring='roc_auc'
)
print(f"Random Forest  5-fold CV AUC: {scores_rf.mean():.3f} ± {scores_rf.std():.3f}")

# 5.3 XGBoost
xgb = XGBClassifier(
    n_estimators=200, max_depth=3, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss', random_state=42
)
scores_xgb = cross_val_score(
    xgb,
    X_train_scaled,
    y_train,
    cv=cv,
    scoring='roc_auc'
)
print(f"XGBoost        5-fold CV AUC: {scores_xgb.mean():.3f} ± {scores_xgb.std():.3f}")

# ----------------------------
# 6. Final model training & evaluation (Logistic Regression chosen)
# ----------------------------
# Fit logistic regression on full training set
logreg_final = LogisticRegression(
    penalty='l2', solver='lbfgs', max_iter=1000, random_state=42
)
logreg_final.fit(X_train_scaled, y_train)

# Predictions on test set
y_pred_proba = logreg_final.predict_proba(X_test_scaled)[:, 1]
y_pred_class = (y_pred_proba > 0.5).astype(int)

# Confusion matrix & metrics
cm = confusion_matrix(y_test, y_pred_class)
tn, fp, fn, tp = cm.ravel()
accuracy = (tp + tn) / (tn + fp + fn + tp)
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
auc_score = roc_auc_score(y_test, y_pred_proba)

print("\n--- Final Logistic Regression Evaluation on TEST SET ---")
print(f"Confusion Matrix:\n{cm}")
print(f"Accuracy   : {accuracy:.3f}")
print(f"Sensitivity: {sensitivity:.3f}")
print(f"Specificity: {specificity:.3f}")
print(f"AUC-ROC    : {auc_score:.3f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_class, digits=3))

# 6.1 ROC Curve data (optional plotting)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
# If desired, one could plot fpr vs. tpr here with matplotlib:
# import matplotlib.pyplot as plt
# plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
# plt.plot([0,1], [0,1], 'k--')
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("ROC Curve (Logistic Regression)")
# plt.legend(loc="lower right")
# plt.tight_layout()
# plt.savefig("roc_logistic_phase3.png")


# ----------------------------
# 7. Multicollinearity check (VIF) for final logistic model
# ----------------------------
X_train_const = sm.add_constant(X_train_scaled)
vif_data = pd.DataFrame({
    'Feature': X_train_const.columns,
    'VIF': [
        variance_inflation_factor(X_train_const.values, i)
        for i in range(X_train_const.shape[1])
    ]
})
print("\nVariance Inflation Factors:")
print(vif_data.round(3))


# ----------------------------
# 8. (Optional) Save final model coefficients
# ----------------------------
coefs = pd.DataFrame({
    'Feature': ['Intercept'] + numeric_cols,
    'Coefficient': np.concatenate(([logreg_final.intercept_[0]], logreg_final.coef_[0]))
})
coefs.to_csv('phase3_logistic_coefficients.csv', index=False)

print("\nPhase 3 processing complete. Coefficients saved to 'phase3_logistic_coefficients.csv'.")




# Save final model
import joblib
joblib.dump(logreg_final, 'final_housing_insecurity_model.pkl')
print("\nFinal model saved as 'final_housing_insecurity_model.pkl'.")