import os
import streamlit as st
import requests
from transformers import pipeline
from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()
api_key = os.getenv("NEWS_API_KEY")

# --- Load Hugging Face models ---
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
sentiment_model = pipeline("sentiment-analysis")

# --- Streamlit UI ---
st.set_page_config(page_title="AI Stock News Impact Analyzer", page_icon="💹", layout="centered")

st.title("💹 AI-Powered Stock News Impact Analyzer")
st.write(
    "Enter a company name below to fetch recent news, summarize each article, "
    "and interpret how it might affect the company's stock price."
)

# --- User Input ---
company_name = st.text_input("Enter a company name (e.g., Apple, Tesla, Nvidia):", "")

if st.button("Analyze") and company_name:
    st.info(f"Fetching latest news for **{company_name}**...")

    # --- Fetch News ---
    url = f"https://newsapi.org/v2/everything?q={company_name}&language=en&sortBy=publishedAt&pageSize=5&apiKey={api_key}"
    response = requests.get(url)
    data = response.json()

    if "articles" in data and len(data["articles"]) > 0:
        st.subheader("🗞️ Latest News Summaries & AI Interpretations")

        for article in data["articles"]:
            title = article.get("title", "Untitled")
            description = article.get("description", "")
            content = (article.get("content") or description or "")[:1500]
            url_link = article.get("url", "")
            source = article.get("source", {}).get("name", "Unknown source")
            published_at = article.get("publishedAt", "")[:10]

            # --- Summarize the article ---
            try:
                summary = summarizer(content, max_length=80, min_length=30, do_sample=False)[0]["summary_text"]
            except Exception:
                summary = description if description else "Summary unavailable."

            # --- Sentiment analysis ---
            try:
                sentiment = sentiment_model(summary)[0]
                label = sentiment["label"]
                conf = round(sentiment["score"], 2)

                if label == "POSITIVE":
                    impact = f"🟢 This news may have a **positive** effect on {company_name}'s stock. (confidence {conf})"
                elif label == "NEGATIVE":
                    impact = f"🔴 This news may have a **negative** effect on {company_name}'s stock. (confidence {conf})"
                else:
                    impact = f"⚪ This news is **neutral**, unlikely to strongly move the stock. (confidence {conf})"
            except Exception:
                impact = "⚪ Unable to evaluate impact."

            # --- Display each article neatly ---
            st.markdown(f"### [{title}]({url_link})")
            st.caption(f"🗓️ {published_at} | 📰 {source}")
            st.markdown(f"**Summary:** {summary}")
            st.markdown(f"**AI Interpretation:** {impact}")
            st.markdown("---")

    else:
        st.error("No recent news found for that company.")
