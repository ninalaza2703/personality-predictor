import numpy as np
import pandas as pd
import re
import string
import spacy
from tqdm import tqdm
import nltk
from textblob import TextBlob
import emoji
from sklearn.model_selection import train_test_split
from pathlib import Path

# Load and Group Reddit MBTI Data

# Load scraped Reddit data
project_root = Path(__file__).resolve().parents[2]
data_path = project_root / "data" / "interim" / "reddit_mbti_chunked.csv"
aggregated_df = pd.read_csv(data_path)
aggregated_df = aggregated_df[aggregated_df["text"].notna()]


# Basic Text Preprocessing

nlp = spacy.load("en_core_web_sm")

# Lowercasing
aggregated_df["posts"] = aggregated_df["text"].str.lower()

# Remove special characters, URLs, emojis, punctuation, and digits
aggregated_df["posts"] = aggregated_df["posts"].astype(str).str.replace(r"[^A-Za-z\s]", "", regex=True)
aggregated_df["posts"] = aggregated_df["posts"].str.replace(r"http\S+|www\S+|https\S+", "", regex=True)
aggregated_df["posts"] = aggregated_df["posts"].str.replace(r"\s+", " ", regex=True).str.strip()
aggregated_df["posts"] = aggregated_df["posts"].apply(lambda x: emoji.replace_emoji(x, replace=""))
aggregated_df["posts"] = aggregated_df["posts"].str.replace(f"[{re.escape(string.punctuation)}]", "", regex=True)
aggregated_df["posts"] = aggregated_df["posts"].str.replace(r"\d+", "", regex=True)
aggregated_df["posts"] = aggregated_df["posts"].str.replace(r"(.)\1{2,}", r"\1\1", regex=True)  # Normalize repeated chars

# Lemmatization with spaCy

cleaned_texts = []
for text in tqdm(aggregated_df["posts"], desc="Lemmatizing"):
    doc = nlp(text)
    lemmatized = [token.lemma_ for token in doc if token.is_alpha and token.lemma_ != "-PRON-"]
    cleaned_texts.append(" ".join(lemmatized))

aggregated_df["posts"] = cleaned_texts


# Feature Engineering

nltk.download("punkt")
nlp = spacy.load("en_core_web_sm")
aggregated_df = aggregated_df.dropna(subset=["posts"])

tqdm.pandas()
tokens_list = []
polarities = []
subjectivities = []

# Tokenization and sentiment analysis
for text in tqdm(aggregated_df["posts"], desc="Tokenizing & analyzing sentiment"):
    tokens = nltk.word_tokenize(text)
    alpha_tokens = [w for w in tokens if w.isalpha()]
    tokens_list.append(alpha_tokens)

    blob = TextBlob(text)
    polarities.append(blob.sentiment.polarity)
    subjectivities.append(blob.sentiment.subjectivity)

# Lexical and sentiment features
aggregated_df["tokens"] = tokens_list
aggregated_df["text_len"] = aggregated_df["tokens"].apply(len)
aggregated_df["avg_word_length"] = aggregated_df["tokens"].apply(
    lambda toks: sum(len(w) for w in toks) / len(toks) if toks else 0
)
aggregated_df["lexical_diversity"] = aggregated_df["tokens"].apply(
    lambda toks: len(set(toks)) / len(toks) if toks else 0
)
aggregated_df["sentiment_polarity"] = polarities
aggregated_df["sentiment_subjectivity"] = subjectivities

# Final Cleanup and Target Encoding

# Remove short posts
aggregated_df = aggregated_df[aggregated_df["text_len"] > 450]
aggregated_df.drop(columns=["tokens", "text_len"], inplace=True)

aggregated_df['type'] = aggregated_df['mbti_from_flair']
# Split MBTI type into four binary traits
aggregated_df["E/I"] = aggregated_df["type"].str[0]
aggregated_df["N/S"] = aggregated_df["type"].str[1]
aggregated_df["F/T"] = aggregated_df["type"].str[2]
aggregated_df["P/J"] = aggregated_df["type"].str[3]

# Binary encoding for each trait
aggregated_df["target_1"] = np.where(aggregated_df["E/I"] == "I", 0, 1)
aggregated_df["target_2"] = np.where(aggregated_df["N/S"] == "N", 0, 1)
aggregated_df["target_3"] = np.where(aggregated_df["F/T"] == "T", 0, 1)
aggregated_df["target_4"] = np.where(aggregated_df["P/J"] == "P", 0, 1)

# Train/Test Split and Save

train_data, test_data = train_test_split(
    aggregated_df,
    test_size=0.2,
    stratify=aggregated_df["target_1"],
    random_state=4
)


processed_dir = project_root / "data" / "processed"

# Create the processed directory if it doesn't exist
processed_dir.mkdir(parents=True, exist_ok=True)
train_data.to_csv(processed_dir / "train_data.csv", index=False)
test_data.to_csv(processed_dir / "test_data.csv", index=False)
