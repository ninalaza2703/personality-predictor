import pandas as pd
import re
import time
import praw
from pathlib import Path
from datetime import datetime

# Initialize Reddit API with PRAW

reddit = praw.Reddit(
    client_id="confidential token",
    client_secret="confidential token",
    user_agent="Scraping"
)

# MBTI subreddits to target
mbti_subreddits = [
    "INTJ", "INTP", "ENTJ", "ENTP", "INFJ", "INFP", "ENFJ", "ENFP",
    "ISTJ", "ISFJ", "ESTJ", "ESFJ", "ISTP", "ISFP", "ESTP", "ESFP"
]

# Regex to find MBTI type claims in post/comment text
pattern = re.compile(
    r"\bI(?:'m| am|’m)? an? (" + "|".join(mbti_subreddits) + r")\b",
    re.IGNORECASE
)

final_user_df = pd.DataFrame()

# Loop through each MBTI subreddit

for mbti_sub in mbti_subreddits:
    print(f"\n### Scraping r/{mbti_sub} ###")

    subreddit = reddit.subreddit(mbti_sub)
    post_generator = subreddit.new(limit=None)

    NUM_ITERATIONS = 100
    BATCH_SIZE = 100

    try:
        for run in range(NUM_ITERATIONS):
            print(f"\n--- RUN {run + 1}/{NUM_ITERATIONS} on r/{mbti_sub} ---")
            data = []

            # Scrape Posts and Comments
            for _ in range(BATCH_SIZE):
                try:
                    post = next(post_generator)
                    data.append({
                        "Type": "Post",
                        "Post_id": post.id,
                        "Title": post.title,
                        "Author": post.author.name if post.author else "Unknown",
                        "Timestamp": post.created_utc,
                        "Text": post.selftext or "",
                        "Score": post.score,
                        "Total_comments": post.num_comments,
                        "Post_URL": post.url
                    })

                    # Include all top-level and nested comments
                    if post.num_comments > 0:
                        post.comments.replace_more(limit=0)
                        for comment in post.comments.list():
                            data.append({
                                "Type": "Comment",
                                "Post_id": post.id,
                                "Title": post.title,
                                "Author": comment.author.name if comment.author else "Unknown",
                                "Timestamp": pd.to_datetime(comment.created_utc, unit="s"),
                                "Text": comment.body or "",
                                "Score": comment.score,
                                "Total_comments": 0,
                                "Post_URL": None
                            })

                    time.sleep(3)

                except StopIteration:
                    print("No more posts in generator.")
                    break
                except Exception as e:
                    print(f"Error processing post/comment: {e}")
                    continue

            df = pd.DataFrame(data)

            # Identify MBTI self-claims
            def extract_mbti(text):
                if pd.isna(text) or not text:
                    return None
                match = pattern.search(text)
                return match.group(1).upper() if match else None

            df["MBTI_claimed"] = df["Text"].apply(extract_mbti)
            claimed_df = df[df["MBTI_claimed"].notnull()]
            claimed_users = dict(zip(claimed_df["Author"], claimed_df["MBTI_claimed"]))
            print(f"Found {len(claimed_users)} MBTI-claiming users")

            # Scrape full history for each claimed user
            user_data = []

            for i, (username, mbti_type) in enumerate(claimed_users.items()):
                try:
                    redditor = reddit.redditor(username)
                    time.sleep(5)

                    # Fetch recent comments
                    for comment in redditor.comments.new(limit=75):
                        user_data.append({
                            "Username": username,
                            "Type": "Comment",
                            "MBTI": mbti_type,
                            "Timestamp": comment.created_utc,
                            "Text": comment.body or "",
                            "Title": "",
                            "Score": comment.score,
                            "Subreddit": comment.subreddit.display_name
                        })

                    # Fetch recent posts
                    for submission in redditor.submissions.new(limit=75):
                        user_data.append({
                            "Username": username,
                            "Type": "Post",
                            "MBTI": mbti_type,
                            "Timestamp": submission.created_utc,
                            "Text": getattr(submission, "selftext", "") or "",
                            "Title": getattr(submission, "title", ""),
                            "Score": submission.score,
                            "Subreddit": submission.subreddit.display_name
                        })

                except Exception as e:
                    print(f"[{i + 1}/{len(claimed_users)}] Failed {username}: {e}")
                    continue

            user_df = pd.DataFrame(user_data)
            final_user_df = pd.concat([final_user_df, user_df], ignore_index=True)
            print(f"Completed run {run + 1} on r/{mbti_sub}, collected {len(user_df)} entries")

            time.sleep(30)

    except Exception as e:
        print(f"\nScript stopped early for r/{mbti_sub} due to error: {e}")

# Save Scraped Dataset

project_root = Path(__file__).resolve().parents[2]
raw_dir = project_root / "data" / "raw"
raw_dir.mkdir(parents=True, exist_ok=True)

output_path = raw_dir / "reddit_mbti_scraped.csv"
final_user_df.to_csv(output_path, index=False)