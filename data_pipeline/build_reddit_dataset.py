from pathlib import Path
import json
import re
import pandas as pd
from tqdm import tqdm



def load_jsonl_file(filepath):
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return data


def load_all_jsonl(data_dir):
    all_records = []
    files = [f for f in Path(data_dir).glob("*.jsonl")]

    for file in tqdm(files, desc="Loading files"):
        all_records.extend(load_jsonl_file(file))

    return pd.DataFrame(all_records)


def extract_and_filter_mbti(df):
    pattern = re.compile(r'(?i)\b([IE][SN][TF][JP])\b')
    df["mbti_from_flair"] = (
        df["author_flair_text"].astype(str).str.extract(pattern, expand=False).str.upper()
    )
    df= df[df["mbti_from_flair"].notna()]
    user_type_counts = df.groupby("author")["mbti_from_flair"].nunique()
    consistent_users = user_type_counts[user_type_counts == 1].index

    df = df[df["author"].isin(consistent_users)].copy()
    df = df[
        df["body"].notna() &
        (df["body"] != "[deleted]") &
        (df["body"] != "[removed]")
    ]
    return df


def chunk_author_stream(
    data_mbti,
    author_col="author",
    time_col="created_utc",
    text_col="body",
    label_col="mbti_from_flair",
    chunk_size=500,
    min_words=450,
):
    df = data_mbti[[author_col, time_col, text_col, label_col]].copy()
    df = df.sort_values([author_col, time_col])
    df[text_col] = df[text_col].astype(str)

    author_text = (
        df.groupby([author_col, label_col], as_index=False)[text_col]
        .apply(lambda x: " ".join(x.tolist()))
        .rename(columns={text_col: "author_text"})
    )

    rows = []
    for _, r in author_text.iterrows():
        words = r["author_text"].split()
        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i + chunk_size]
            if len(chunk_words) < min_words:
                continue
            rows.append({
                author_col: r[author_col],
                label_col: r[label_col],
                "chunk_idx": i // chunk_size,
                "text": " ".join(chunk_words),
                "n_words": len(chunk_words),
            })

    return pd.DataFrame(rows)



project_root = Path(__file__).resolve().parents[2]
data_dir = project_root / "data" / "raw_bulk"
interim_dir = project_root / "data" / "interim"
interim_dir.mkdir(parents=True, exist_ok=True)

df = load_all_jsonl(data_dir)
df = extract_and_filter_mbti(df)
aggregated_df = chunk_author_stream(df)

aggregated_df.to_csv(interim_dir / "reddit_mbti_chunked.csv", index=False)