import os
import json
import time
import random
import requests
from datetime import datetime, timezone

SUBREDDITS = [
        "istj", "isfj", "istp", "isfp",
    "estj", "esfj", "estp", "esfp",
    # Core MBTI
    "mbti",
    "mbtimemes",
    "MbtiTypeMe",

    # 16 personality types
    "intj", "intp", "infj", "infp",
    "entj", 
    "entp",
    "enfj", 
    "enfp",


    # Typology & personality theory
    "personality",
    "enneagram",
    "typology",
    "cognitivefunctions",
    "jung",
    "jungiantypology",
    "Socionics"
]

OUT_DIR = "data/raw_bulk"
os.makedirs(OUT_DIR, exist_ok=True)

API_URL = "https://api.pullpush.io/reddit/search/comment/"

def to_ts(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp())

START_TS = to_ts("2020-01-01")
END_TS = int(datetime.now(timezone.utc).timestamp())

BATCH_SIZE = 500
BASE_SLEEP = 2.0
MAX_EMPTY_PAGES = 5
MAX_RETRIES = 20

TIMEOUT = 30


HEAVY_SUBS = {"mbti", "personality"}
USE_MONTH_WINDOWS_FOR_HEAVY = True
USE_YEAR_WINDOWS_FOR_OTHERS = True

GLOBAL_COOLDOWN_SEC = 10 * 60  # 10 minutes


# Window generators

def year_ranges(start_year=2020, end_ts=END_TS):
    now_year = datetime.fromtimestamp(end_ts, tz=timezone.utc).year
    for y in range(start_year, now_year + 1):
        a = int(datetime(y, 1, 1, tzinfo=timezone.utc).timestamp())
        b = int(datetime(y + 1, 1, 1, tzinfo=timezone.utc).timestamp()) if y < now_year else end_ts
        yield a, b, f"{y}"

def month_ranges(start_year=2020, end_ts=END_TS):
    end_dt = datetime.fromtimestamp(end_ts, tz=timezone.utc)
    y, m = start_year, 1
    while (y < end_dt.year) or (y == end_dt.year and m <= end_dt.month):
        a = int(datetime(y, m, 1, tzinfo=timezone.utc).timestamp())
        if m == 12:
            b_dt = datetime(y + 1, 1, 1, tzinfo=timezone.utc)
        else:
            b_dt = datetime(y, m + 1, 1, tzinfo=timezone.utc)
        b = min(int(b_dt.timestamp()), end_ts)
        yield a, b, f"{y}-{m:02d}"
        m += 1
        if m == 13:
            m = 1
            y += 1


# Checkpoint helpers

def checkpoint_path(subreddit: str, window_label: str):
    return os.path.join(OUT_DIR, f".checkpoint_{subreddit}_{window_label}.json")

def load_checkpoint(subreddit: str, window_label: str, default_after: int):
    path = checkpoint_path(subreddit, window_label)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
            return int(d.get("after", default_after)), int(d.get("n_total", 0))
        except Exception:
            return default_after, 0
    return default_after, 0

def save_checkpoint(subreddit: str, window_label: str, after: int, n_total: int):
    path = checkpoint_path(subreddit, window_label)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"after": int(after), "n_total": int(n_total)}, f)
    os.replace(tmp, path)

def out_path_for(subreddit: str, window_label: str):
    # append mode to continue if rerun
    return os.path.join(OUT_DIR, f"comments_{subreddit}_{window_label}.jsonl")


# Rate-limit handling

class HardRateLimit(Exception):
    """Raised when we keep getting 429 too many times in a row."""
    pass

def safe_get_json(session: requests.Session, params: dict):
    """
    Request with exponential backoff + jitter on 429/5xx.
    If 429 persists for MAX_RETRIES attempts, raise HardRateLimit.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(API_URL, params=params, timeout=TIMEOUT)

            if r.status_code == 429:
                ra = r.headers.get("Retry-After")
                if ra is not None:
                    wait = float(ra)
                else:
                    wait = min(120.0, (2 ** (attempt - 1)) * 1.5)

                wait = wait + random.uniform(0, 1.0)
                print(f"429 rate-limited. Sleeping {wait:.1f}s (attempt {attempt}/{MAX_RETRIES})...")
                time.sleep(wait)
                continue

            if 500 <= r.status_code < 600:
                wait = min(60.0, (2 ** (attempt - 1)) * 1.0) + random.uniform(0, 1.0)
                print(f"{r.status_code} server error. Sleeping {wait:.1f}s (attempt {attempt}/{MAX_RETRIES})...")
                time.sleep(wait)
                continue

            r.raise_for_status()
            return r.json()

        except requests.exceptions.RequestException as e:
            wait = min(60.0, (2 ** (attempt - 1)) * 1.0) + random.uniform(0, 1.0)
            print(f"Request error: {e}. Sleeping {wait:.1f}s (attempt {attempt}/{MAX_RETRIES})...")
            time.sleep(wait)

    # successful response
    raise HardRateLimit("Hard rate limit: too many consecutive 429s / request failures.")


def fetch_comments(session: requests.Session, subreddit: str, after_ts: int, before_ts: int):
    params = {
        "subreddit": subreddit,
        "after": after_ts,
        "before": before_ts,
        "size": BATCH_SIZE,
        "sort": "asc",
        "sort_type": "created_utc",
    }
    j = safe_get_json(session, params)
    return j.get("data", [])


# Main scraping logic

def windows_for_sub(subreddit: str):
    # Heavy subs: monthly windows
    if USE_MONTH_WINDOWS_FOR_HEAVY and subreddit.lower() in {s.lower() for s in HEAVY_SUBS}:
        return list(month_ranges(2020, END_TS))
    # Others: yearly windows
    if USE_YEAR_WINDOWS_FOR_OTHERS:
        return list(year_ranges(2020, END_TS))
    # Fallback: one big window
    return [(START_TS, END_TS, "2020plus")]


session = requests.Session()
session.headers.update({
    "User-Agent": "mbti-reddit-research/1.0 (contact: nina)"
})

skipped = []

for sub in SUBREDDITS:
    print(f"Starting subreddit: r/{sub}")

    windows = windows_for_sub(sub)

    for after_window, before_window, label in windows:
        out_path = out_path_for(sub, label)
        print(f"\n--- Window {label}: {datetime.fromtimestamp(after_window, tz=timezone.utc).date()} "
              f"→ {datetime.fromtimestamp(before_window, tz=timezone.utc).date()} ---")
        print(f"Output: {out_path}")

        after, n_total = load_checkpoint(sub, label, after_window)
        empty_pages = 0
        last_progress_print = time.time()

        with open(out_path, "a", encoding="utf-8") as f:
            while after < before_window:
                try:
                    data = fetch_comments(session, sub, after, before_window)
                except HardRateLimit as e:
                    print(f"[SKIP WINDOW] {e}")
                    print(f"Skipping r/{sub} window {label} for now.")
                    save_checkpoint(sub, label, after, n_total)
                    skipped.append((sub, label))
                    print(f"Global cooldown {GLOBAL_COOLDOWN_SEC/60:.0f} minutes to reduce chance of immediate re-lock...")
                    time.sleep(GLOBAL_COOLDOWN_SEC)
                    break  # move to next window/subreddit
                except Exception as e:
                    # Non-429 related error: wait briefly and keep trying
                    print(f"Unexpected error: {e}. Sleeping 30s and retrying...")
                    time.sleep(30)
                    continue

                if not data:
                    empty_pages += 1
                    if empty_pages >= MAX_EMPTY_PAGES:
                        print("No more data in this window. Moving on.")
                        save_checkpoint(sub, label, after, n_total)
                        break
                    time.sleep(BASE_SLEEP)
                    continue

                empty_pages = 0

                # write batch
                for c in data:
                    f.write(json.dumps(c, ensure_ascii=False) + "\n")
                n_total += len(data)

                # paginate: advance after timestamp
                last_ts = data[-1].get("created_utc", after)
                after = max(after + 1, int(last_ts) + 1)

                # checkpoint every batch
                save_checkpoint(sub, label, after, n_total)

                now = time.time()
                if (n_total % 5000) < BATCH_SIZE or (now - last_progress_print) > 10:
                    dt = datetime.fromtimestamp(after, tz=timezone.utc).strftime("%Y-%m-%d")
                    print(f"Downloaded {n_total:,} comments in this window. Now at ~{dt} UTC")
                    last_progress_print = now

                time.sleep(BASE_SLEEP)

        print(f"Done window {label} for r/{sub}: {n_total:,} comments")

print("\n==============================")
print("SCRAPE FINISHED (for now)")
print("==============================")
if skipped:
    print("Skipped windows due to hard rate limit (you can rerun later to continue):")
    for sub, label in skipped:
        print(f" - r/{sub} window {label}")
else:
    print("No windows were skipped.")