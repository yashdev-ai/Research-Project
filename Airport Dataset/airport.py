import pandas as pd

# Load your Excel file
file_path = "airport_dataset.xlsv"
df = pd.read_excel(file_path, sheet_name='combined_data_airport')

# Clean column names (remove extra spaces)
df.columns = df.columns.str.strip()

# Check if 'Comments' column exists
if 'text' not in df.columns:
    raise KeyError("❌ Column 'Comments' not found in the dataset. Available columns are:\n" + str(df.columns))

# Function to evaluate health impact
def evaluate_health_impact(comment):
    if pd.isna(comment) or comment.strip() == "":
        return "No", "No comment provided"
    
    comment_lower = comment.lower()

    health_indicators = [
        "fatigue", "tired", "health issue", "not feeling well", "unwell", "exhausted", 
        "heat", "hot", "temperature", "noise", "sweating", "dizzy", "headache", "ill", 
        "sick", "dehydrated", "noisy", "humid", "overheated", "ear pain"
    ]

    if any(word in comment_lower for word in health_indicators):
        return "Yes", "Comment indicates health impact due to temperature or noise"
    else:
        return "No", "No mention of health impact due to temperature or noise"

# Apply function
df['Relevance'] = df['text'].apply(lambda x: evaluate_health_impact(x)[0])
df['Remarks'] = df['text'].apply(lambda x: evaluate_health_impact(x)[1])

# Save the modified file
df.to_excel("airport_dataset_with_relevance.xlsx", index=False)

print("✅ Columns added and file saved as 'airport_dataset_with_relevance.xlsx'")
