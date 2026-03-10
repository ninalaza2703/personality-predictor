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
data_path = project_root / "data" / "raw" / "reddit_mbti_scraped.csv"
data_mbti = pd.read_csv(data_path)
data_mbti = data_mbti.sort_values("Username").reset_index(drop=True)
data_mbti = data_mbti[data_mbti["Text"].notna()]

# Group every 20 posts per user as a single block
data_mbti["GroupID"] = data_mbti.groupby("Username").cumcount() // 20
aggregated_df = (
    data_mbti.groupby(["Username", "GroupID", "MBTI"])["Text"]
    .apply(lambda x: " ".join(x))
    .reset_index()
)

# Basic Text Preprocessing

nlp = spacy.load("en_core_web_sm")

# Lowercasing
aggregated_df["posts"] = aggregated_df["Text"].str.lower()

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

# Merge with Existing MBTI Dataset

mbti_old = pd.read_csv("data/raw/MBTI 500.csv")
aggregated_df.rename(columns={"MBTI": "type"}, inplace=True)
aggregated_df["scraped"] = 1

all_data = pd.concat([mbti_old, aggregated_df[["type", "posts", "scraped"]]])
all_data["scraped"].fillna(0, inplace=True)

# Feature Engineering

nltk.download("punkt")
nlp = spacy.load("en_core_web_sm")
all_data = all_data.dropna(subset=["posts"])

tqdm.pandas()
tokens_list = []
polarities = []
subjectivities = []

# Tokenization and sentiment analysis
for text in tqdm(all_data["posts"], desc="Tokenizing & analyzing sentiment"):
    tokens = nltk.word_tokenize(text)
    alpha_tokens = [w for w in tokens if w.isalpha()]
    tokens_list.append(alpha_tokens)

    blob = TextBlob(text)
    polarities.append(blob.sentiment.polarity)
    subjectivities.append(blob.sentiment.subjectivity)

# Lexical and sentiment features
all_data["tokens"] = tokens_list
all_data["text_len"] = all_data["tokens"].apply(len)
all_data["avg_word_length"] = all_data["tokens"].apply(
    lambda toks: sum(len(w) for w in toks) / len(toks) if toks else 0
)
all_data["lexical_diversity"] = all_data["tokens"].apply(
    lambda toks: len(set(toks)) / len(toks) if toks else 0
)
all_data["sentiment_polarity"] = polarities
all_data["sentiment_subjectivity"] = subjectivities

# Final Cleanup and Target Encoding

# Remove short posts
all_data = all_data[all_data["text_len"] > 16]
all_data.drop(columns=["tokens", "text_len"], inplace=True)

# Split MBTI type into four binary traits
all_data["E/I"] = all_data["type"].str[0]
all_data["N/S"] = all_data["type"].str[1]
all_data["F/T"] = all_data["type"].str[2]
all_data["P/J"] = all_data["type"].str[3]

# Binary encoding for each trait
all_data["target_1"] = np.where(all_data["E/I"] == "I", 0, 1)
all_data["target_2"] = np.where(all_data["N/S"] == "N", 0, 1)
all_data["target_3"] = np.where(all_data["F/T"] == "T", 0, 1)
all_data["target_4"] = np.where(all_data["P/J"] == "P", 0, 1)

# Train/Test Split and Save

train_data, test_data = train_test_split(
    all_data,
    test_size=0.2,
    stratify=all_data["target_1"],
    random_state=4
)


project_root = Path(__file__).resolve().parents[2]
processed_dir = project_root / "data" / "processed"

# Create the processed directory if it doesn't exist
processed_dir.mkdir(parents=True, exist_ok=True)
train_data.to_csv(processed_dir / "train_data.csv", index=False)
test_data.to_csv(processed_dir / "test_data.csv", index=False)
