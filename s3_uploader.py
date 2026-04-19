"""
S3 Uploader - Phase 2
Uploads HN sentiment data to AWS S3 bucket
- Raw JSON data per cycle
- Daily summary reports
- CSV exports
"""

import boto3
import json
import os
import psycopg2
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BUCKET_NAME = os.getenv("S3_BUCKET", "hn-sentiment-data-ashish")
AWS_REGION  = os.getenv("AWS_REGION", "us-east-1")

DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     os.getenv("DB_PORT", 5432),
    "dbname":   os.getenv("DB_NAME", "reddit_sentiment"),
    "user":     os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
}


def get_s3_client():
    return boto3.client(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )


def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)


def upload_raw_json(s3, posts: list[dict]):
    """Upload raw post data as JSON — organised by date/hour."""
    now = datetime.utcnow()
    key = f"raw/{now.strftime('%Y/%m/%d')}/hn_posts_{now.strftime('%H%M%S')}.json"

    payload = {
        "fetched_at": now.isoformat(),
        "total_posts": len(posts),
        "posts": posts,
    }

    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=key,
        Body=json.dumps(payload, indent=2, default=str),
        ContentType="application/json",
    )
    logger.info(f"  Uploaded raw JSON → s3://{BUCKET_NAME}/{key}")
    return key


def upload_daily_summary(s3):
    """Pull today's data from PostgreSQL and upload summary CSV to S3."""
    conn = get_db_connection()
    query = """
        SELECT subreddit, sentiment_label,
               COUNT(*) as post_count,
               ROUND(AVG(vader_compound)::numeric, 4) as avg_vader,
               ROUND(AVG(score)::numeric, 1) as avg_score
        FROM reddit_posts
        WHERE fetched_at >= NOW() - INTERVAL '24 hours'
        GROUP BY subreddit, sentiment_label
        ORDER BY subreddit, sentiment_label
    """
    df = pd.read_sql(query, conn)
    conn.close()

    now = datetime.utcnow()
    key = f"summaries/{now.strftime('%Y/%m/%d')}/daily_summary_{now.strftime('%H%M%S')}.csv"

    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=key,
        Body=df.to_csv(index=False),
        ContentType="text/csv",
    )
    logger.info(f"  Uploaded daily summary → s3://{BUCKET_NAME}/{key}")
    return key


def upload_full_export(s3):
    """Upload full database export as CSV to S3."""
    conn = get_db_connection()
    df = pd.read_sql("SELECT * FROM reddit_posts ORDER BY fetched_at DESC", conn)
    conn.close()

    now = datetime.utcnow()
    key = f"exports/full_export_{now.strftime('%Y%m%d_%H%M%S')}.csv"

    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=key,
        Body=df.to_csv(index=False),
        ContentType="text/csv",
    )
    logger.info(f"  Uploaded full export → s3://{BUCKET_NAME}/{key}")
    return key


def list_s3_files(s3, prefix=""):
    """List all files in the bucket."""
    response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)
    files = []
    for obj in response.get("Contents", []):
        files.append({
            "key":           obj["Key"],
            "size_kb":       round(obj["Size"] / 1024, 2),
            "last_modified": obj["LastModified"].strftime("%Y-%m-%d %H:%M:%S"),
        })
    return files


def run_s3_backup():
    """Run a full backup cycle to S3."""
    logger.info("Starting S3 backup...")
    s3 = get_s3_client()

    # Test connection
    s3.head_bucket(Bucket=BUCKET_NAME)
    logger.info(f"Connected to S3 bucket: {BUCKET_NAME}")

    upload_daily_summary(s3)
    upload_full_export(s3)

    files = list_s3_files(s3)
    logger.info(f"\nBucket contents ({len(files)} files):")
    for f in files:
        logger.info(f"  {f['key']} ({f['size_kb']} KB)")

    logger.info("S3 backup complete!")


if __name__ == "__main__":
    run_s3_backup()
