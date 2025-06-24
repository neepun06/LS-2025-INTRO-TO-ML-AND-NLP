import pandas as pd
import numpy as np

# Sample data
data = {
    'Name': [f'Student{i}' for i in range(1, 11)],
    'Subject': np.random.choice(['Math', 'Science', 'English'], 10),
    'Score': np.random.randint(50, 101, 10),
    'Grade': [''] * 10
}

df = pd.DataFrame(data)

# Assign grades
def assign_grade(score):
    if score >= 90: return 'A'
    elif score >= 80: return 'B'
    elif score >= 70: return 'C'
    elif score >= 60: return 'D'
    else: return 'F'

df['Grade'] = df['Score'].apply(assign_grade)

# Print sorted by Score
print("Sorted by Score:\n", df.sort_values(by='Score', ascending=False))

# Average score per subject
print("Average Score per Subject:\n", df.groupby('Subject')['Score'].mean())

# Filter pass function
def pandas_filter_pass(dataframe):
    return dataframe[dataframe['Grade'].isin(['A', 'B'])]

print("Filtered Pass Records:\n", pandas_filter_pass(df))
