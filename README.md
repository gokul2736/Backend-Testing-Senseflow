# Backend-Testing-Senseflow

## Raw Data sets Used
<img width="442" height="464" alt="image" src="https://github.com/user-attachments/assets/d939ab9e-57c2-4d6e-a98a-a5ee304d4e3b" />

## Combined and made into Single Data set
i.e "processed_data (1)" csv file
https://drive.google.com/file/d/1zQcKkvW-xz7KLIDmyODZC8W72FyBdyYo/view?usp=sharing

issue *
<img width="925" height="199" alt="image" src="https://github.com/user-attachments/assets/a92ff83b-d58b-4c67-a35b-bfb3566440f6" />

## 
```python3
import pandas as pd
import re
import html

# 1. Inspect the dataset
print("--- STEP 1: INITIAL INSPECTION ---")
df = pd.read_csv("processed_data (1).csv", on_bad_lines='skip')
print(f"Dataset Shape: {df.shape}")
print(f"Column Names: {list(df.columns)}")
print("\nFirst 10 rows:")
print(df.head(10))

# 2. Check data quality
print("\n--- STEP 2: QUALITY AUDIT ---")
print(f"Missing Values:\n{df.isnull().sum()}")
print(f"Duplicate Rows: {df.duplicated().sum()}")
print(f"Label Distribution:\n{df['label'].value_counts()}")

# 3. Clean the dataset
def advanced_clean(text):
    if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
        return None

    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = html.unescape(text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s.,!?$@]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    if text == "":
        return None
    return text

print("\n--- STEP 3: CLEANING DATASET ---")
# Using 'clean_text' as the source column based on your previous data structure
df['text'] = df['clean_text'].apply(advanced_clean)

# Remove rows with missing email text
df = df.dropna(subset=['text'])

# Remove duplicate emails
df = df.drop_duplicates(subset=['text'])

# 4. Verify labels
print("\n--- STEP 4: LABEL VERIFICATION ---")
# Confirm labels are binary (0 = safe, 1 = phishing)
df = df[df['label'].isin([0, 1, 0.0, 1.0])].copy()
df['label'] = df['label'].astype(int)
print(f"Rows for Safe (0): {len(df[df['label'] == 0])}")
print(f"Rows for Phishing (1): {len(df[df['label'] == 1])}")

# 5. Generate a final cleaned dataset
final_df = df[['text', 'label']]
final_df.to_csv('senseflow_clean_dataset.csv', index=False)
print("\n✅ SUCCESS: senseflow_clean_dataset.csv generated successfully.")

# Final dataset analysis
print("\n--- STEP 5: FINAL DATASET ANALYSIS ---")
print("Average text length:")
print(final_df['text'].apply(len).describe())
```
<img width="556" height="786" alt="image" src="https://github.com/user-attachments/assets/067392ed-2df7-498a-a59c-c7864fe9feaf" />

## removes that extra words and other stuff....
```python 
import pandas as pd

# Load the clean data
df = pd.read_csv('senseflow_clean_dataset.csv')

# The AI can only read about 512 words at a time anyway. 
# This chops every email down to the first 512 words.
df['text'] = df['text'].apply(lambda x: ' '.join(str(x).split()[:512]))

# Save the final, ML-ready file
df.to_csv('senseflow_ready_for_ai.csv', index=False)

print("✅ Monster emails destroyed. Data is officially ready for training.")
```

<img width="830" height="286" alt="image" src="https://github.com/user-attachments/assets/8383299f-c32f-4e5a-9282-29583dd1b873" />

<img width="784" height="564" alt="image" src="https://github.com/user-attachments/assets/dcf20c17-acb3-4338-885f-6a0cbaa80e74" />
<img width="874" height="373" alt="image" src="https://github.com/user-attachments/assets/fb4c27a5-f0f4-4112-9af1-e8821ae66f94" />

SenseFlow Data Preprocessing Report
1. Initial State of the Data

Input File: processed_data (1).csv

Starting Shape: 16,568 rows.

Condition: Partially labeled (0 = Safe, 1 = Phishing) but contained heavy web noise, duplicates, and severe outliers.

2. Challenges Faced & Solutions

Challenge 1: Web Noise & Formatting Junk. The text was full of HTML tags (<br>, <div>) and URLs that distract the AI from reading the actual human intent.

How we fixed it: We used Python re (regex) and html libraries to strip URLs and unescape HTML entities.

Challenge 2: Preserving "Panic" Context. Standard cleaning removes all special characters, but hackers use specific symbols for financial manipulation.

How we fixed it: We customized the regex filter re.sub(r'[^a-zA-Z0-9\s.,!?$@]', '', text) to strictly preserve $, @, !, and ? so the model can still detect urgency and money requests.

Challenge 3: Duplicates and Blanks. We found completely empty rows and exact copies that would cause the model to overfit.

How we fixed it: We used pandas to apply dropna(subset=['text']) and drop_duplicates(subset=['text']). We removed 11 blanks and 273 exact duplicates.

Challenge 4: The "Monster" Outlier (Memory Crash Risk). A .describe() audit revealed one row was 4.29 million characters long (likely a corrupted log file). Feeding this to a T4 GPU would trigger an Out-Of-Memory (OOM) crash.

How we fixed it: We applied a strict length filter df[df['text'].apply(lambda x: len(str(x)) < 50000)]. This eradicated 17 corrupted "monster" rows.

3. Final Results & Output

Output File: senseflow_clean_dataset.csv

Final Shape: 16,135 clean rows.

Class Distribution: 11,523 Phishing (1) | 4,612 Safe (0).

Text Length Stats: The max length dropped from 4.2 million characters down to a safe 46,332 characters. The average email is now 1,274 characters long, which is the perfect sweet spot for phishing detection.

Status: Deterministic cleaning is complete. The dataset is structurally sound and ready for Hugging Face tokenization and DistilBERT fine-tuning


<img width="685" height="698" alt="image" src="https://github.com/user-attachments/assets/42bc55f4-2da2-40ad-b918-2b31328cb320" />















```
