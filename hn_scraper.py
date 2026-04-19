"""
Hacker News Sentiment Analysis Pipeline
Fetches real live stories from Hacker News API (no key needed!)
-> Sentiment Analysis (VADER + TextBlob)
-> PostgreSQL Storage
"""

import psycopg2
from psycopg2.extras import execute_batch
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
import time
import logging
import os
import requests
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     os.getenv("DB_PORT", 5432),
    "dbname":   os.getenv("DB_NAME", "reddit_sentiment"),
    "user":     os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
}

HN_BASE = "https://hacker-news.firebaseio.com/v0"

# Categories we'll pull from
CATEGORIES = {
    "topstories":  "Top Stories",
    "newstories":  "New Stories",
    "beststories": "Best Stories",
    "askstories":  "Ask HN",
    "showstories": "Show HN",
}

vader = SentimentIntensityAnalyzer()


def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)


def analyze_sentiment(text: str) -> dict:
    vader_scores = vader.polarity_scores(text)
    vader_compound = vader_scores["compound"]
    blob = TextBlob(text)

    if vader_compound >= 0.05:
        label = "positive"
    elif vader_compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"

    return {
        "vader_compound":  round(vader_compound, 4),
        "vader_positive":  round(vader_scores["pos"], 4),
        "vader_negative":  round(vader_scores["neg"], 4),
        "vader_neutral":   round(vader_scores["neu"], 4),
        "tb_polarity":     round(blob.sentiment.polarity, 4),
        "tb_subjectivity": round(blob.sentiment.subjectivity, 4),
        "sentiment_label": label,
    }


def fetch_story_ids(category: str, limit: int = 30) -> list:
    url = f"{HN_BASE}/{category}.json"
    resp = requests.get(url, timeout=10)
    ids = resp.json()
    return ids[:limit]


def fetch_story(story_id: int) -> dict | None:
    url = f"{HN_BASE}/item/{story_id}.json"
    resp = requests.get(url, timeout=10)
    data = resp.json()
    if not data or data.get("type") != "story" or not data.get("title"):
        return None
    return data


def fetch_category_posts(category: str, limit: int = 30) -> list[dict]:
    posts = []
    story_ids = fetch_story_ids(category, limit)

    for story_id in story_ids:
        try:
            story = fetch_story(story_id)
            if not story:
                continue

            title = story.get("title", "")
            text  = story.get("text", "") or ""
            full_text = f"{title} {text}".strip()
            sentiment = analyze_sentiment(full_text)

            posts.append({
                "post_id":      str(story_id),
                "subreddit":    category,           # reusing subreddit column for category
                "title":        title[:500],
                "text":         text[:2000],
                "author":       story.get("by", "[unknown]"),
                "score":        story.get("score", 0),
                "num_comments": story.get("descendants", 0),
                "upvote_ratio": 1.0,               # HN doesn't have downvotes
                "url":          story.get("url", f"https://news.ycombinator.com/item?id={story_id}"),
                "created_utc":  datetime.utcfromtimestamp(story.get("time", time.time())),
                "fetched_at":   datetime.utcnow(),
                **sentiment,
            })
            time.sleep(0.05)   # be polite to HN API

        except Exception as e:
            logger.warning(f"  Skipping story {story_id}: {e}")
            continue

    logger.info(f"  [{category}]: fetched {len(posts)} stories")
    return posts


def save_to_db(conn, posts: list[dict]):
    if not posts:
        return
    query = """
        INSERT INTO reddit_posts (
            post_id, subreddit, title, text, author,
            score, num_comments, upvote_ratio, url,
            created_utc, fetched_at,
            vader_compound, vader_positive, vader_negative, vader_neutral,
            tb_polarity, tb_subjectivity, sentiment_label
        ) VALUES (
            %(post_id)s, %(subreddit)s, %(title)s, %(text)s, %(author)s,
            %(score)s, %(num_comments)s, %(upvote_ratio)s, %(url)s,
            %(created_utc)s, %(fetched_at)s,
            %(vader_compound)s, %(vader_positive)s, %(vader_negative)s, %(vader_neutral)s,
            %(tb_polarity)s, %(tb_subjectivity)s, %(sentiment_label)s
        )
        ON CONFLICT (post_id) DO UPDATE SET
            score           = EXCLUDED.score,
            num_comments    = EXCLUDED.num_comments,
            fetched_at      = EXCLUDED.fetched_at,
            vader_compound  = EXCLUDED.vader_compound,
            sentiment_label = EXCLUDED.sentiment_label;
    """
    with conn.cursor() as cur:
        execute_batch(cur, query, posts)
    conn.commit()
    logger.info(f"  Saved {len(posts)} stories to DB")


def run_pipeline(interval_seconds: int = 300):
    logger.info("Hacker News Sentiment Pipeline started (no API key needed!)")
    conn = get_db_connection()
    cycle = 0

    try:
        while True:
            cycle += 1
            logger.info(f"\n{'─'*50}")
            logger.info(f"Cycle #{cycle} at {datetime.utcnow().strftime('%H:%M:%S')} UTC")

            for category in CATEGORIES:
                try:
                    posts = fetch_category_posts(category, limit=20)
                    save_to_db(conn, posts)
                except Exception as e:
                    logger.error(f"Error on {category}: {e}")
                time.sleep(2)

            logger.info(f"Cycle #{cycle} done. Next run in {interval_seconds}s...")
            time.sleep(interval_seconds)

    except KeyboardInterrupt:
        logger.info("Pipeline stopped.")
    finally:
        conn.close()


if __name__ == "__main__":
    run_pipeline(interval_seconds=300)


# ─── S3 Integration (called from run_pipeline) ───────────────────────────────
def upload_cycle_to_s3(posts: list[dict]):
    """Upload this cycle's posts to S3. Silently skips if S3 not configured."""
    try:
        import boto3, json, os
        from datetime import datetime

        bucket = os.getenv("S3_BUCKET")
        key_id = os.getenv("AWS_ACCESS_KEY_ID")
        secret = os.getenv("AWS_SECRET_ACCESS_KEY")

        if not all([bucket, key_id, secret]):
            return  # S3 not configured, skip silently

        s3 = boto3.client(
            "s3",
            region_name=os.getenv("AWS_REGION", "us-east-1"),
            aws_access_key_id=key_id,
            aws_secret_access_key=secret,
        )
        now = datetime.utcnow()
        key = f"raw/{now.strftime('%Y/%m/%d')}/hn_posts_{now.strftime('%H%M%S')}.json"
        payload = {"fetched_at": now.isoformat(), "total_posts": len(posts), "posts": posts}
        s3.put_object(
            Bucket=bucket, Key=key,
            Body=json.dumps(payload, indent=2, default=str),
            ContentType="application/json",
        )
        logger.info(f"  ☁️  Uploaded {len(posts)} posts to S3")
    except Exception as e:
        logger.warning(f"  S3 upload skipped: {e}")
