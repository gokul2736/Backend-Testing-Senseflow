# The SenseFlow Data Architecture

## Raw Data sets Used
<img width="442" height="464" alt="image" src="https://github.com/user-attachments/assets/d939ab9e-57c2-4d6e-a98a-a5ee304d4e3b" />

## Phase 1: Data Ingestion & Normalization
### Objective: Aggregate scattered raw datasets into a unified, single-schema Data Lake.

* Automated Ingestion: Acted as a "data vacuum," automatically pulling from multiple distinct sources (e.g., Nigerian_Fraud.csv, SpamAssasin.csv, synthetic data).

* Dynamic Normalization: Auto-detected varying column names across different files and forced them into a strict binary schema (text, label).

* Output: Successfully merged disparate data into processed_data.csv, establishing our initial raw dataset of **16,568 emails**.
### Phase 1 Script
```python3
import pandas as pd
import re
import os
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

def build_dynamic_cloud_data_lake():
    print("--- SenseFlow: Initiating Dynamic Cloud Data Pipeline ---")
    dfs = []

    # Colab default upload directory is the current working directory ('.')
    for file in os.listdir('.'):
        try:
            if file.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(file, engine='openpyxl')
            else:
                continue # Skip folders or other file types

            # Auto-detect columns
            text_col = 'body' if 'body' in df.columns else 'content' if 'content' in df.columns else 'text' if 'text' in df.columns else None
            label_col = 'label' if 'label' in df.columns else 'Class' if 'Class' in df.columns else None

            if text_col and label_col:
                temp_df = df[[text_col, label_col]].rename(columns={text_col: 'text', label_col: 'label'})
                dfs.append(temp_df)
                print(f"✅ Successfully ingested: {file}")
            else:
                print(f"⚠️ Skipped {file}: Could not find 'body/content' and 'label' columns.")
        except Exception as e:
            print(f"❌ Error loading {file}: {e}")

    if not dfs:
        print("CRITICAL ERROR: No valid data found to merge.")
        return

    print("\nMerging datasets and purging duplicates...")
    master_df = pd.concat(dfs, ignore_index=True).drop_duplicates().dropna()

    def sanitize_text(text):
        text = str(text).lower()
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
        text = re.sub(r'http\S+', '[URL]', text)
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text.strip()

    print(f"Sanitizing {len(master_df)} emails... (Please wait)")
    master_df['clean_text'] = master_df['text'].apply(sanitize_text)
    master_df['label'] = pd.to_numeric(master_df['label'], errors='coerce').fillna(0).astype(int)

    output_path = 'processed_data.csv'
    master_df[['clean_text', 'label']].to_csv(output_path, index=False)
    print(f"\n🎉 SUCCESS! Final Data Lake generated: {output_path}")
    print(f"Total Unique Training Samples: {len(master_df)}")

build_dynamic_cloud_data_lake()
```

## Result of P1 Data Cleaning
<img width="867" height="401" alt="image" src="https://github.com/user-attachments/assets/64c8db78-07ba-461f-a844-f984e4782f8f" />


## Combined and made into Single Data set
i.e  [processed_data](https://drive.google.com/file/d/1zQcKkvW-xz7KLIDmyODZC8W72FyBdyYo/view?usp=sharing) **csv file** 



## Phase 2 - Deep Sanitization & EDA
### Objective: Sterilize the raw data and eliminate hardware-crashing anomalies to make it ML-ready.

* Targeted Noise Reduction: Stripped HTML tags and web links using Regex, while intentionally preserving high-signal "panic symbols" ($, @, !) used in phishing.

* Anomaly & OOM Prevention: Purged 273 exact duplicates to prevent model overfitting, and eradicated a critical 4.2-million-character corrupted log file that would have triggered a GPU Out-Of-Memory (OOM) crash.

* Output: Generated senseflow_clean_dataset.csv, resulting in 16,135 pristine, researcher-grade samples perfectly optimized for Transformer training.

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

## Result of P2 Data Cleaning
<img width="556" height="786" alt="image" src="https://github.com/user-attachments/assets/37913fc8-ca46-4f40-a173-f7ad711ad6c7" />


## Phase 2B Intervention
***A mutated email that was 4.2 million characters long.***

<img width="376" height="243" alt="image" src="https://github.com/user-attachments/assets/ec408c2b-2b99-4209-9802-5560faaf08db" />   
A fatal dataset flaw discovered during EDA to ensure pipeline and hardware stability. 
We implemented an automated length-filtering gatekeeper (len < 50,000) that scans every row and permanently purges oversized
```python3
import pandas as pd

# 1. Load the dataset we just cleaned
df = pd.read_csv('senseflow_clean_dataset.csv')

# 2. Automated Security Gate: Purge corrupted records (>50k characters)
df = df[df['text'].apply(lambda x: len(str(x)) < 50000)]

# 3. Overwrite the file with the truly clean data
df.to_csv('senseflow_clean_dataset.csv', index=False)

print("✅ Anomalies eradicated. Let's look at the safe stats:")
print(df['text'].apply(lambda x: len(str(x))).describe())
```

* Before Phase 2b Filter: Max text length: 4,296,597 characters

* After Phase 2b Filter: Max text length: 46,332 characters

***What this code does:*** It acts as an automated security checkpoint. It scans the length of every single row. If it finds a row over 50,000 characters, it assumes it is a corrupted log file or a giant block of base64 image code, and it permanently deletes it from the CSV.

The "Why" (Hardware Protection): If you did not run this code, that 4.2-million-character row would have been passed into Phase 3. When the Tokenizer tried to translate 4 million characters at once, it would have overloaded your Colab's RAM and caused a fatal Out-Of-Memory (OOM) crash.


The Issue Discovered: After initial text sanitization, our Exploratory Data Analysis (EDA) revealed a critical hidden anomaly: a single mutated record containing over 4.29 million characters (identified as a corrupted server log/base64 dump).

The Infrastructure Risk: We calculated that passing a multi-million-character sequence to the Hugging Face Tokenizer would immediately exceed the Colab T4 GPU's memory limits, guaranteeing a fatal Out-Of-Memory (OOM) crash during training.

The Phase 2b Solution: To solve this, we engineered a dedicated anomaly detection step. , unprocessable artifacts before they can reach the AI model.

<img width="925" height="199" alt="image" src="https://github.com/user-attachments/assets/a92ff83b-d58b-4c67-a35b-bfb3566440f6" />
The Critical Anomaly & Phase 2b Intervention
Objective: Overcome a fatal dataset flaw discovered during EDA to ensure pipeline and hardware stability.

The Issue Discovered: After initial text sanitization, our Exploratory Data Analysis (EDA) revealed a critical hidden anomaly: a single mutated record containing over 4.29 million characters (identified as a corrupted server log/base64 dump).

The Infrastructure Risk: We calculated that passing a multi-million-character sequence to the Hugging Face Tokenizer would immediately exceed the Colab T4 GPU's memory limits, guaranteeing a fatal Out-Of-Memory (OOM) crash during training.

The Phase 2b Solution: To solve this, we engineered a dedicated anomaly detection step. We implemented an automated length-filtering gatekeeper (len < 50,000) that scans every row and permanently purges oversized, unprocessable artifacts before they can reach the AI model.

## 2B Filter Verification
```python3
import pandas as pd

print("--- POST-PHASE 2b VERIFICATION REPORT ---")
# 1. Load the final, clean dataset
df = pd.read_csv('senseflow_clean_dataset.csv')

# 2. Print the final row count to ensure we didn't delete too much
print(f"Total safe rows remaining: {len(df)}")

# 3. Print the length stats to PROVE the 4-million-character monster is dead
print("\nFinal Text Length Statistics:")
print(df['text'].apply(lambda x: len(str(x))).describe())
```
<img width="491" height="293" alt="image" src="https://github.com/user-attachments/assets/6d3d2140-6e46-4d97-b000-21938266deeb" />



## SenseFlow Data Preprocessing Report
### 1. Initial State of the Data
Input File: processed_data (1).csv
Starting Shape: 16,568 rows.
Condition: Partially labeled (0 = Safe, 1 = Phishing) but contained heavy web noise, duplicates, and severe outliers.



<img width="387" height="534" alt="image" src="https://github.com/user-attachments/assets/e24dc6cf-d12d-4779-9404-21188d09e326" />

<img width="560" height="141" alt="image" src="https://github.com/user-attachments/assets/21794144-0a82-4777-b80d-2a90f0d574ab" />




## 2. Challenges Faced & Solutions

### Challenge 1: Web Noise & Formatting Junk. The text was full of HTML tags (<br>, <div>) and URLs that distract the AI from reading the actual human intent.
How we fixed it: We used Python re (regex) and html libraries to strip URLs and unescape HTML entities.

### Challenge 2: Preserving "Panic" Context. Standard cleaning removes all special characters, but hackers use specific symbols for financial manipulation.

How we fixed it: We customized the regex filter re.sub(r'[^a-zA-Z0-9\s.,!?$@]', '', text) to strictly preserve $, @, !, and ? so the model can still detect urgency and money requests.

### Challenge 3: Duplicates and Blanks. We found completely empty rows and exact copies that would cause the model to overfit.

How we fixed it: We used pandas to apply dropna(subset=['text']) and drop_duplicates(subset=['text']). We removed 11 blanks and 273 exact duplicates.

### Challenge 4: The "Monster" Outlier (Memory Crash Risk). A .describe() audit revealed one row was 4.29 million characters long (likely a corrupted log file). Feeding this to a T4 GPU would trigger an Out-Of-Memory (OOM) crash.

How we fixed it: We applied a strict length filter ```df[df['text'].apply(lambda x: len(str(x)) < 50000)]```. This eradicated 17 corrupted "monster" rows.























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

<img width="873" height="608" alt="image" src="https://github.com/user-attachments/assets/86eca7b2-be34-4ec7-88a8-9ca62230f7d9" />


# Precaution 
<img width="933" height="543" alt="image" src="https://github.com/user-attachments/assets/6d8a5d3f-d1ab-4ef2-8568-484519afc9ff" />

<img width="651" height="213" alt="image" src="https://github.com/user-attachments/assets/12d000d6-30a1-4d71-b7c1-7daac8b3af3d" />


## what I Have done 


Combined multiple Kaggle datasets
✔ Built a deterministic preprocessing pipeline
✔ Cleaned noise, duplicates, and corrupted rows
✔ Generated senseflow_clean_dataset.csv (16,135 rows)
✔ Verified the dataset structure


<img width="389" height="140" alt="image" src="https://github.com/user-attachments/assets/229d3c8e-246a-482f-9a2e-6f280cea2c8f" />



we moved from 512 to 256

<img width="901" height="566" alt="image" src="https://github.com/user-attachments/assets/1313aafd-35e4-4e95-9d3c-9fac458da577" />


<img width="877" height="480" alt="image" src="https://github.com/user-attachments/assets/44ab439f-eccf-4cd3-85b3-8bcb8fbdab8d" />


<img width="904" height="442" alt="image" src="https://github.com/user-attachments/assets/1728eadc-728a-4ea3-8f26-cea622efbb1f" />

<img width="752" height="574" alt="image" src="https://github.com/user-attachments/assets/0855e40a-59d9-4d60-9da6-3f90e2feff08" />
```import pandas as pd
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







```

The SenseFlow Presentation Flow
1. The Starting Point (Phase 1: Data Ingestion)

What you show: The Phase 1 slide.

What you say: "To train DistilBERT effectively, we needed a diverse dataset. For Phase 1, we built an automated ingestion pipeline that aggregated multiple distinct threat-intelligence feeds into a single, unified Data Lake, giving us an initial raw dataset of over 16,500 emails."

2. The Cleanup (Phase 2a: Deep Sanitization)

What you show: The Phase 2a slide (HTML/Regex cleaning).

What you say: "Raw data is too noisy for machine learning. In Phase 2a, we applied deterministic regex filtering to strip out HTML artifacts and URLs. Crucially, we preserved high-signal symbols like '$' and '@' which are strong indicators of phishing intent. We also purged exact duplicates to prevent the model from memorizing repetitive data."

3. The Climax (Phase 2b: Hardware Protection)

What you show: The Phase 2b slide (The < 50000 filter).

What you say: "This is where we caught a critical infrastructure risk. During our Exploratory Data Analysis, we discovered a fatal anomaly: a single corrupted record containing 4.29 million characters. If passed to the Hugging Face tokenizer, this would have instantly caused a Colab T4 GPU Out-Of-Memory crash. To guarantee hardware stability, we engineered a Phase 2b automated gatekeeper that permanently purges any sequence over 50,000 characters."

4. The Proof (Final Analytics Report)

What you show: The screenshot of your final analytics (the 16,135 rows, the max length of 46,332).

What you say: "Here is the validation of our pipeline. We successfully reduced the maximum sequence length to a safe 46,000 characters, leaving us with a pristine dataset of exactly 16,135 samples. The class balance is roughly 71% phishing to 28% safe, which accurately mirrors real-world threat environments where spam heavily outweighs legitimate mail."

```
