"""
Hacker News Sentiment Dashboard — Streamlit
Run: streamlit run dashboard.py
"""

import streamlit as st
import psycopg2
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="HN Sentiment Pipeline",
    page_icon="🔶",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stMetric > label { font-size: 0.8rem; color: #94a3b8; }
    div[data-testid="metric-container"] { background: #1e1e2e; border-radius: 10px; padding: 12px; }
</style>
""", unsafe_allow_html=True)

CATEGORY_LABELS = {
    "topstories":  "🔥 Top Stories",
    "newstories":  "🆕 New Stories",
    "beststories": "⭐ Best Stories",
    "askstories":  "❓ Ask HN",
    "showstories": "🚀 Show HN",
}

@st.cache_resource
def get_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", 5432),
        dbname=os.getenv("DB_NAME", "reddit_sentiment"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", ""),
    )

@st.cache_data(ttl=60)
def load_data(hours_back=24, categories=None):
    conn = get_connection()
    since = datetime.utcnow() - timedelta(hours=hours_back)
    cat_filter = ""
    params = [since]
    if categories:
        placeholders = ",".join(["%s"] * len(categories))
        cat_filter = f"AND subreddit IN ({placeholders})"
        params += categories

    query = f"""
        SELECT post_id, subreddit, title, author, score, num_comments,
               created_utc, fetched_at,
               vader_compound, vader_positive, vader_negative, vader_neutral,
               tb_polarity, tb_subjectivity, sentiment_label, url
        FROM reddit_posts
        WHERE fetched_at >= %s {cat_filter}
        ORDER BY fetched_at DESC
        LIMIT 5000
    """
    df = pd.read_sql(query, conn, params=params)
    df["category_label"] = df["subreddit"].map(CATEGORY_LABELS).fillna(df["subreddit"])
    return df

# ─── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Filters")
hours_back = st.sidebar.slider("Time window (hours)", 1, 72, 24)
auto_refresh = st.sidebar.checkbox("Auto-refresh (60s)", value=False)
if auto_refresh:
    time.sleep(60)
    st.rerun()

try:
    conn = get_connection()
    all_cats = pd.read_sql("SELECT DISTINCT subreddit FROM reddit_posts ORDER BY subreddit", conn)["subreddit"].tolist()
except Exception:
    all_cats = list(CATEGORY_LABELS.keys())

selected = st.sidebar.multiselect(
    "HN Categories",
    options=all_cats,
    default=all_cats,
    format_func=lambda x: CATEGORY_LABELS.get(x, x)
)

# ─── Load Data ───────────────────────────────────────────────────────────────
try:
    df = load_data(hours_back, selected if selected else None)
    data_ok = len(df) > 0
except Exception as e:
    st.error(f"DB connection failed: {e}")
    st.info("Make sure PostgreSQL is running and .env is configured correctly.")
    st.stop()

# ─── Header ──────────────────────────────────────────────────────────────────
st.title("🔶 Hacker News Sentiment Analysis Pipeline")
st.caption(f"Live data from news.ycombinator.com  |  Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC  |  {len(df):,} stories loaded")

if not data_ok:
    st.warning("No data yet. Run the pipeline first: `python hn_scraper.py`")
    st.stop()

# ─── KPI Row ─────────────────────────────────────────────────────────────────
total = len(df)
pos   = (df["sentiment_label"] == "positive").sum()
neg   = (df["sentiment_label"] == "negative").sum()
neu   = (df["sentiment_label"] == "neutral").sum()
avg_vader = df["vader_compound"].mean()
avg_score = int(df["score"].mean())

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Total Stories",  f"{total:,}")
c2.metric("Positive 😊",    f"{pos:,}",  f"{pos/total*100:.1f}%")
c3.metric("Neutral 😐",     f"{neu:,}",  f"{neu/total*100:.1f}%")
c4.metric("Negative 😞",    f"{neg:,}",  f"{neg/total*100:.1f}%")
c5.metric("Avg VADER",      f"{avg_vader:.3f}")
c6.metric("Avg Points",     f"{avg_score:,}")

st.divider()

# ─── Charts Row 1 ────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Sentiment by Category")
    pivot = df.groupby(["category_label", "sentiment_label"]).size().reset_index(name="count")
    fig = px.bar(
        pivot, x="category_label", y="count", color="sentiment_label", barmode="group",
        color_discrete_map={"positive": "#4ade80", "neutral": "#94a3b8", "negative": "#f87171"},
        template="plotly_dark", labels={"category_label": "Category", "count": "Stories"},
    )
    fig.update_layout(margin=dict(t=10, b=10), legend_title="Sentiment")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Average VADER Score by Category")
    avg_by_cat = df.groupby("category_label")["vader_compound"].mean().reset_index()
    fig2 = px.bar(
        avg_by_cat, x="category_label", y="vader_compound",
        color="vader_compound", color_continuous_scale=["#f87171", "#94a3b8", "#4ade80"],
        range_color=[-1, 1], template="plotly_dark",
        labels={"category_label": "Category", "vader_compound": "VADER Score"},
    )
    fig2.add_hline(y=0, line_dash="dot", line_color="white", opacity=0.4)
    fig2.update_layout(margin=dict(t=10, b=10), coloraxis_showscale=False)
    st.plotly_chart(fig2, use_container_width=True)

# ─── Charts Row 2 ────────────────────────────────────────────────────────────
col3, col4 = st.columns(2)

with col3:
    st.subheader("Sentiment Trend Over Time")
    df["hour"] = pd.to_datetime(df["fetched_at"]).dt.floor("H")
    time_df = df.groupby(["hour", "sentiment_label"]).size().reset_index(name="count")
    fig3 = px.line(
        time_df, x="hour", y="count", color="sentiment_label",
        color_discrete_map={"positive": "#4ade80", "neutral": "#94a3b8", "negative": "#f87171"},
        template="plotly_dark",
    )
    fig3.update_layout(margin=dict(t=10, b=10))
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    st.subheader("VADER vs TextBlob Polarity")
    sample = df.sample(min(500, len(df)))
    fig4 = px.scatter(
        sample, x="vader_compound", y="tb_polarity",
        color="sentiment_label", size="score",
        hover_data=["category_label", "title"],
        color_discrete_map={"positive": "#4ade80", "neutral": "#94a3b8", "negative": "#f87171"},
        template="plotly_dark", opacity=0.7,
        labels={"vader_compound": "VADER Score", "tb_polarity": "TextBlob Polarity"},
    )
    fig4.add_hline(y=0, line_dash="dot", line_color="white", opacity=0.3)
    fig4.add_vline(x=0, line_dash="dot", line_color="white", opacity=0.3)
    fig4.update_layout(margin=dict(t=10, b=10))
    st.plotly_chart(fig4, use_container_width=True)

# ─── Sentiment Donut ─────────────────────────────────────────────────────────
col5, col6 = st.columns(2)

with col5:
    st.subheader("Overall Sentiment Split")
    pie_df = df["sentiment_label"].value_counts().reset_index()
    pie_df.columns = ["sentiment", "count"]
    fig5 = px.pie(
        pie_df, names="sentiment", values="count", hole=0.5,
        color="sentiment",
        color_discrete_map={"positive": "#4ade80", "neutral": "#94a3b8", "negative": "#f87171"},
        template="plotly_dark",
    )
    fig5.update_layout(margin=dict(t=10, b=10))
    st.plotly_chart(fig5, use_container_width=True)

with col6:
    st.subheader("Top Scoring Stories")
    top = df.nlargest(10, "score")[["title", "score", "sentiment_label", "category_label"]]
    st.dataframe(top, use_container_width=True, height=300, hide_index=True)

# ─── Raw Data Table ───────────────────────────────────────────────────────────
st.divider()
st.subheader("📋 Latest Stories")
display_cols = ["category_label", "title", "sentiment_label", "vader_compound", "score", "num_comments", "fetched_at"]
st.dataframe(df[display_cols].head(100), use_container_width=True, height=400, hide_index=True)

st.download_button(
    "⬇️ Export CSV",
    df.to_csv(index=False).encode(),
    f"hn_sentiment_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
    "text/csv",
)
