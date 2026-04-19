-- ============================================================
-- Reddit Sentiment Pipeline — Database Schema
-- Run once: psql -U postgres -d reddit_sentiment -f schema.sql
-- ============================================================

-- Create database (run as superuser if it doesn't exist)
-- CREATE DATABASE reddit_sentiment;

CREATE TABLE IF NOT EXISTS reddit_posts (
    id               SERIAL PRIMARY KEY,
    post_id          VARCHAR(20)   UNIQUE NOT NULL,       -- Reddit post ID
    subreddit        VARCHAR(100)  NOT NULL,
    title            TEXT          NOT NULL,
    text             TEXT          DEFAULT '',
    author           VARCHAR(100),
    score            INTEGER       DEFAULT 0,
    num_comments     INTEGER       DEFAULT 0,
    upvote_ratio     FLOAT,
    url              TEXT,
    created_utc      TIMESTAMP     NOT NULL,
    fetched_at       TIMESTAMP     NOT NULL DEFAULT NOW(),

    -- VADER scores
    vader_compound   FLOAT,
    vader_positive   FLOAT,
    vader_negative   FLOAT,
    vader_neutral    FLOAT,

    -- TextBlob scores
    tb_polarity      FLOAT,
    tb_subjectivity  FLOAT,

    -- Final label
    sentiment_label  VARCHAR(10)   CHECK (sentiment_label IN ('positive','negative','neutral'))
);

-- Indexes for dashboard queries
CREATE INDEX IF NOT EXISTS idx_subreddit      ON reddit_posts (subreddit);
CREATE INDEX IF NOT EXISTS idx_sentiment      ON reddit_posts (sentiment_label);
CREATE INDEX IF NOT EXISTS idx_fetched_at     ON reddit_posts (fetched_at DESC);
CREATE INDEX IF NOT EXISTS idx_created_utc    ON reddit_posts (created_utc DESC);
CREATE INDEX IF NOT EXISTS idx_vader_compound ON reddit_posts (vader_compound);

-- ─── Aggregated view for dashboards ───────────────────────────────────────────
CREATE OR REPLACE VIEW sentiment_summary AS
SELECT
    subreddit,
    sentiment_label,
    COUNT(*)                          AS post_count,
    ROUND(AVG(vader_compound)::numeric, 4) AS avg_vader,
    ROUND(AVG(score)::numeric, 1)     AS avg_score,
    ROUND(AVG(num_comments)::numeric, 1) AS avg_comments,
    MAX(fetched_at)                   AS last_updated
FROM reddit_posts
GROUP BY subreddit, sentiment_label
ORDER BY subreddit, sentiment_label;

-- ─── Hourly trend view ────────────────────────────────────────────────────────
CREATE OR REPLACE VIEW hourly_sentiment AS
SELECT
    DATE_TRUNC('hour', fetched_at)    AS hour,
    subreddit,
    sentiment_label,
    COUNT(*)                          AS post_count,
    ROUND(AVG(vader_compound)::numeric, 4) AS avg_vader
FROM reddit_posts
GROUP BY 1, 2, 3
ORDER BY 1 DESC;

-- Sample check query
-- SELECT * FROM sentiment_summary;
-- SELECT * FROM hourly_sentiment LIMIT 20;
