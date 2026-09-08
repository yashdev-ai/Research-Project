# Import required libraries
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.discrete.discrete_model import Logit

import warnings
from statsmodels.tools.sm_exceptions import HessianInversionWarning

warnings.filterwarnings("ignore", category=HessianInversionWarning)


# Load the dataset
df = pd.read_excel('housing_insecurity_prediction.xlsx', sheet_name='cleaned')

# Correct the problematic column name and verify others
cols_to_drop = [
    'Start Date', 'End Date', 'Response Type', 'IP Address', 'Progress',
    'Duration (in seconds)', 'Finished', 'Recorded Date', 'Response ID',
    'Recipient Last Name', 'Recipient First Name', 'Location Latitude',
    'Location Longitude', 'Distribution Channel', 'User Language',
    'E-mail (this e-mail will be used to distribute your giftcard)',
    'Please provide the name of the community college you attended.',
    # Use the FULL column name below:
    'We would like to follow-up with you on your responses with an interview. The interview will take a maximum of 30-45 minutes. As a result of COVID-19, we will conduct interviews via a teleconference tool.  For your participation in the interview, you will receive a $25 gift-card to Wal-Mart. Are you willing to participate in a follow-up interview?'
]

# Before dropping, check if columns exist
missing_cols = [col for col in cols_to_drop if col not in df.columns]
if missing_cols:
    print(f"Warning: Columns not found and will be skipped: {missing_cols}")
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]

df = df.drop(columns=cols_to_drop)

# Preprocess multi-choice columns into dummy variables
def split_multichoice_columns(df, column_name):
    dummies = df[column_name].str.get_dummies(', ')
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(columns=[column_name])
    return df

# Apply to relevant columns
df = split_multichoice_columns(df, 'How do you usually describe your race and/or ethnicity?')
df = split_multichoice_columns(df, 'Which of the following ways do you pay for the expenses associated with attending college? (check all that apply)')

# Convert Yes/No columns to binary (1/0)
yes_no_cols = [
    'In the past 12 months, did you couch surf – that is, moved from one temporary housing arrangement to another because you had no other place to live?',
    'Have you ever been in foster care?',
    # Add other Yes/No columns here
]
for col in yes_no_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0}).fillna(0)

# Define target variable (Couch Surfing)
target = 'In the past 12 months, did you couch surf – that is, moved from one temporary housing arrangement to another because you had no other place to live?'
y = df[target]
X = df.drop(columns=[target])



# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)




# Stepwise feature selection using AIC
def stepwise_selection(X, y, threshold_in=0.05, threshold_out=0.10):
    included = []
    while True:
        changed = False
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded, dtype=float)
        for new_column in excluded:
            try:
                model = sm.Logit(y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))).fit(
                    maxiter=1000, 
                    method='bfgs',
                    disp=0
                )
                new_pval[new_column] = model.pvalues[new_column]
            except:
                continue  # Skip columns causing errors
        if new_pval.empty:
            break
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed = True
        # Backward step
        if included:
            model = sm.Logit(y, sm.add_constant(pd.DataFrame(X[included]))).fit(
                maxiter=1000, 
                method='bfgs',
                disp=0
            )
            pvalues = model.pvalues.iloc[1:]
            worst_pval = pvalues.max()
            if worst_pval > threshold_out:
                worst_feature = pvalues.idxmax()
                included.remove(worst_feature)
                changed = True
        if not changed:
            break
    return included

selected_features = stepwise_selection(X_train, y_train)
print(f'Selected features: {selected_features}')

# Fit final logistic regression model
X_train_final = sm.add_constant(X_train[selected_features])
model = sm.Logit(y_train, X_train_final).fit()
print(model.summary())

# Calculate odds ratios and confidence intervals
odds_ratios = pd.DataFrame({
    'OR': np.exp(model.params),
    'Lower CI': np.exp(model.conf_int()[0]),
    'Upper CI': np.exp(model.conf_int()[1]),
    'p-value': model.pvalues
})
print(odds_ratios)

# Evaluate model performance
X_test_final = sm.add_constant(X_test[selected_features])
y_pred = model.predict(X_test_final)
y_pred_class = (y_pred > 0.5).astype(int)

# Confusion matrix and metrics
cm = confusion_matrix(y_test, y_pred_class)
tn, fp, fn, tp = cm.ravel()
accuracy = (tp + tn) / (tp + tn + fp + fn)
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
auc = roc_auc_score(y_test, y_pred)

print(f'Confusion Matrix:\n{cm}')
print(f'Accuracy: {accuracy:.2f}')
print(f'Sensitivity: {sensitivity:.2f}')
print(f'Specificity: {specificity:.2f}')
print(f'AUC-ROC: {auc:.2f}')

# Check for multicollinearity using VIF
vif_data = pd.DataFrame()
vif_data["Feature"] = X_train_final.columns
vif_data["VIF"] = [variance_inflation_factor(X_train_final.values, i) for i in range(X_train_final.shape[1])]
print(vif_data)