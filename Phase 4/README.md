# Phase 4

Final phase of the housing insecurity prediction project: a trained logistic regression model and a command-line script for generating predictions.

## Files

- `main.py` — final entry-point script. Prompts the user for survey responses, scales them, and predicts the probability of couch surfing / housing insecurity using the trained model.
- `final_housing_insecurity_model.pkl` — trained logistic regression model, loaded by `main.py`.
- `Predicting Housing Insecurity Among Community College Students Using Multivariable Logistic Regression.docx` — write-up of the modeling approach and results.

## Usage

Run from inside this folder so the relative path to the model file resolves:

```bash
cd "Phase 4"
python main.py
```

Answer the prompted Yes/No and numeric questions; the script prints the predicted probability and risk class.

## Requirements

- `numpy`
- `pandas`
- `joblib`
- `scikit-learn`
