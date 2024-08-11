import pandas as pd
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("ruslanmv/ai-medical-chatbot")

# Extract necessary columns and save them to a TSV file
df = pd.DataFrame({
    'text': dataset['train']['Doctor']
})
df.to_csv("ai_medical_chatbot_passages.tsv", sep="\t", index=False, header=False)
