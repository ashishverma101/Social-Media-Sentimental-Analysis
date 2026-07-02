# HN Sentiment Analysis Pipeline

This project analyzes Hacker News posts and performs sentiment analysis.

## Tech Stack
- Python
- Streamlit
- PostgreSQL
- Pandas

## How to Run
1. Install requirements
2. Run : Python hn_scraper.py
3. Run: streamlit run dashboard.py
4. Run : Python s3_uploader.py

## CSV Dashboard Mode

The Streamlit dashboard also accepts a CSV upload from the sidebar, so you can
preview sentiment charts without a running PostgreSQL pipeline. Uploaded files
can use `title`, `text`, `tweet`, `tweet_text`, `full_text`, `content`,
`comment`, `message`, or `post` as the text column.

If `sentiment_label`, `vader_compound`, or `tb_polarity` are missing, the
dashboard computes them locally before rendering the charts.
