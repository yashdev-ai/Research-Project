import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# Define feature names
feature_names = [
    "Have you ever been in foster care?",
    "In the past 12 months, how many times have you moved?",
    "White or Caucasian",
    "Southeast Asian",
    "Hispanic or Latino",
    "Student loans, Help from family or friends",
    "Other grants from the federal or state government, Other grants from my college or university",
    "A work-study job,Savings"
]

# Manually define training scaler params (replace with your real training means and stds!)
feature_means = np.array([0.1, 1.5, 0.3, 0.05, 0.12, 0.45, 0.33, 0.28])
feature_stds = np.array([0.3, 1.2, 0.45, 0.22, 0.32, 0.50, 0.47, 0.42])

# Create a scaler object fitted with those stats
scaler = StandardScaler()
scaler.mean_ = feature_means
scaler.scale_ = feature_stds
scaler.var_ = feature_stds ** 2  # required for some versions of sklearn
scaler.n_features_in_ = len(feature_names)

# Load the model
model = joblib.load("final_housing_insecurity_model.pkl")

# Collect user input
def binary_input(prompt):
    while True:
        val = input(prompt + " (Yes/No): ").strip().lower()
        if val == 'yes': return 1
        if val == 'no': return 0
        print("Please enter Yes or No.")

def numeric_input(prompt):
    while True:
        try:
            return float(input(prompt + " (Enter a number): "))
        except:
            print("Invalid input. Please enter a numeric value.")

print("\nAnswer the following questions:")
user_input = [
    binary_input("Have you ever been in foster care?"),
    numeric_input("How many times have you moved in the past 12 months?"),
    binary_input("Do you identify as White or Caucasian?"),
    binary_input("Do you identify as Southeast Asian?"),
    binary_input("Do you identify as Hispanic or Latino?"),
    binary_input("Do you rely on Student loans or Help from family/friends?"),
    binary_input("Do you receive grants from federal/state or college?"),
    binary_input("Do you use work-study job or savings?")
]

# Convert to DataFrame for model, NumPy array for scaler
user_df = pd.DataFrame([user_input], columns=feature_names)
user_scaled_np = scaler.transform(np.array(user_input).reshape(1, -1))

# Predict
proba = model.predict_proba(user_scaled_np)[0][1]
pred = model.predict(user_scaled_np)[0]

print("\n--- Prediction Result ---")
print(f"Predicted Probability of Couch Surfing: {proba:.3f}")
print(f"Predicted Class: {pred} ({'Likely at risk' if pred == 1 else 'Not at risk'})")
print("\nThank you for your responses!")
print("Please consult a professional for personalized advice.")