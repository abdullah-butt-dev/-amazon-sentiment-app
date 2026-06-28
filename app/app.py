import streamlit as st
import os
import pickle
import pandas as pd
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import plotly.express as px
import plotly.graph_objects as go

nltk.download('vader_lexicon', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)


# ─── PAGE CONFIG ────────────────────────────────────────────────
st.set_page_config(
    page_title="Amazon Review Sentiment Analyzer",
    page_icon="🔍",
    layout="wide"
)

# ─── STYLES ─────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'IBM Plex Mono', monospace;
    }
    .main { background-color: #ffffff; }
    .metric-card {
        background: #f8f8f8;
        border: 1px solid #e0e0e0;
        border-radius: 4px;
        padding: 20px;
        text-align: center;
    }
    .positive { border-left: 4px solid #2ecc71; }
    .negative { border-left: 4px solid #e74c3c; }
    .neutral  { border-left: 4px solid #f39c12; }
    .sentiment-badge {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.4rem;
        font-weight: 600;
        padding: 8px 20px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)


# ─── LOAD MODELS ────────────────────────────────────────────────
@st.cache_resource
def load_models():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    tfidf_path = os.path.join(BASE_DIR, 'tfidf_vectorizer.pkl')
    lr_path = os.path.join(BASE_DIR, 'lr_model.pkl')
    
    with open(tfidf_path, 'rb') as f:
        tfidf = pickle.load(f)
    with open(lr_path, 'rb') as f:
        lr = pickle.load(f)
        
    return tfidf, lr


tfidf, lr = load_models()
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

# ─── HELPER FUNCTIONS ───────────────────────────────────────────
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    return ' '.join(tokens)

def analyze_review(text):
    cleaned = clean_text(text)
    vader = sia.polarity_scores(text)
    tb = TextBlob(cleaned)
    tfidf_vec = tfidf.transform([cleaned])
    lr_pred = lr.predict(tfidf_vec)[0]
    lr_proba = lr.predict_proba(tfidf_vec)[0]
    lr_confidence = max(lr_proba)
    return {
        'vader_compound': vader['compound'],
        'vader_pos': vader['pos'],
        'vader_neg': vader['neg'],
        'vader_neu': vader['neu'],
        'tb_polarity': tb.sentiment.polarity,
        'tb_subjectivity': tb.sentiment.subjectivity,
        'lr_prediction': lr_pred,
        'lr_confidence': lr_confidence,
        'lr_proba': dict(zip(lr.classes_, lr_proba))
    }

def sentiment_label(compound):
    if compound >= 0.05: return 'Positive'
    elif compound <= -0.05: return 'Negative'
    else: return 'Neutral'

def sentiment_color(label):
    return {'Positive': '#2ecc71', 'Negative': '#e74c3c', 'Neutral': '#f39c12'}.get(label, '#999')

# ─── HEADER ─────────────────────────────────────────────────────
st.markdown("# 🔍 Amazon Review Sentiment Analyzer")
st.markdown("*Compares TextBlob, VADER, and Logistic Regression on any review text.*")

# 💡 Simple Explanation Paragraph for Non-Technical Users
st.markdown("""
This tool uses artificial intelligence to detect whether a customer's review text is **Positive**, **Negative**, or **Neutral**. 
Instead of trusting just one system, it runs three distinct AI methods side-by-side so you can see where they agree or disagree.
""")

# 💡 This toggle opens AND closes seamlessly on click
show_info = st.checkbox("ℹ️ Click here to see how these three AI methods work", value=False)

if show_info:
    st.markdown("""
    ### 👋 A Quick Guide to Our Three AI Methods
    
    *   **VADER** is built specifically for internet talk. It acts like an automated checklist, looking for exact words, capital letters for yelling, and exclamation points to guess the mood. It is fantastic at capturing the casual vibe of short online reviews.
    
    *   **TextBlob** takes a more traditional approach. It reads through the text like a digital dictionary, hunting for descriptive adjectives (like "excellent" or "flimsy") and averaging their emotional weights to see if the customer sounds happy or upset.
    
    *   **Our Custom ML Model** is the smartest of the three because it didn't use a pre-made dictionary. Instead, it learned directly from thousands of actual Amazon reviews. By studying real data, it figured out exactly which specific phrases point to a hidden complaint or a genuine compliment.
    """)

st.divider()

# ─── TABS ───────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Single Review Analysis", "Bulk CSV Analysis"])

# ── TAB 1: SINGLE REVIEW ────────────────────────────────────────
with tab1:
    # 💡 NEW: Sample Review Selector Dropdown Menu
    st.markdown("**💡 Try a Sample Review:** Select a preset scenario below to instantly populate the text box without needing to copy-paste or search.")
    sample_selection = st.selectbox(
        "Choose a test scenario to try out:",
        [
            "Type your own custom review...",
            "Test 1: Sarcasm Test",
            "Test 2: Negation Test",
            "Test 3: Mixed Emotion Test",
            "Test 4: Modern Slang Test"
        ]
    )

    # Convert selection to text strings
    default_text = ""
    if "Test 1" in sample_selection:
        default_text = "Oh fantastic, the item arrived broken and two weeks late. Just what I wanted."
    elif "Test 2" in sample_selection:
        default_text = "The build quality is not bad at all, and I do not hate the design."
    elif "Test 3" in sample_selection:
        default_text = "The customer service was absolutely terrible and a total waste of time, but the product itself actually works perfectly."
    elif "Test 4" in sample_selection:
        default_text = "This gadget is pure fire. Legit the best purchase I made this year, zero cap."

    st.markdown("### Paste a review")
    review_text = st.text_area(
        label="Review text",
        value=default_text, # 💡 Connected selection value directly here
        placeholder="Paste any Amazon product review here...",
        height=150,
        label_visibility='collapsed'
    )

    if st.button("Analyze", type="primary"):
        if review_text.strip():
            result = analyze_review(review_text)

            vader_label = sentiment_label(result['vader_compound'])
            tb_label = sentiment_label(result['tb_polarity'])
            lr_label = result['lr_prediction']

            st.markdown("### Results")
            col1, col2, col3 = st.columns(3)

            with col1:
                color = sentiment_color(vader_label)
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size:0.8rem;color:#666;margin-bottom:8px">VADER</div>
                    <div class="sentiment-badge" style="color:{color}">{vader_label}</div>
                    <div style="font-size:0.85rem;color:#888;margin-top:8px">
                        Compound: {result['vader_compound']:.3f}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.caption("🔍 Positive if score is above 0.05. Negative if below -0.05.")

            with col2:
                color = sentiment_color(tb_label)
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size:0.8rem;color:#666;margin-bottom:8px">TextBlob</div>
                    <div class="sentiment-badge" style="color:{color}">{tb_label}</div>
                    <div style="font-size:0.85rem;color:#888;margin-top:8px">
                        Polarity: {result['tb_polarity']:.3f}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.caption("🔍 Scores range from -1 (very unhappy) to +1 (very happy).")

            with col3:
                color = sentiment_color(lr_label)
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size:0.8rem;color:#666;margin-bottom:8px">ML Model (LR)</div>
                    <div class="sentiment-badge" style="color:{color}">{lr_label}</div>
                    <div style="font-size:0.85rem;color:#888;margin-top:8px">
                        Confidence: {result['lr_confidence']:.1%}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.caption("🔍 Shows how certain the model is about its final answer choice.")

            st.markdown("### Confidence Breakdown (ML Model)")
            proba_df = pd.DataFrame([result['lr_proba']]).T.reset_index()
            proba_df.columns = ['Sentiment', 'Probability']
            proba_df['Probability'] = proba_df['Probability'] * 100

            fig = px.bar(proba_df, x='Sentiment', y='Probability',
                color='Sentiment',
                color_discrete_map={'Positive': '#2ecc71', 'Negative': '#e74c3c'},
                title='Model Confidence by Class',
                labels={'Probability': 'Confidence (%)'},
                text=proba_df['Probability'].apply(lambda x: f'{x:.1f}%')
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(template='plotly_white', showlegend=False,
                yaxis_range=[0, 110], height=350)
            st.plotly_chart(fig, use_container_width=True)

            # 💡 Simple, direct chart explanation for non-technical users
            st.info("**What this graph means:** This shows the percentage breakdown of our custom model's guess. A tall, single bar means the AI is highly confident. If the bars look close or even, it means the review contains mixed emotions or tricky sarcasm that is hard to separate.")

            with st.expander("Debug details"):
                st.write("🔧 This panel shows the raw behind-the-scenes decimal math calculated by each model. Regular users can ignore this, but developers use these exact numbers to troubleshoot or fine-tune the AI system.")
                st.json({
                    'VADER scores': {
                        'positive': result['vader_pos'],
                        'negative': result['vader_neg'],
                        'neutral': result['vader_neu'],
                        'compound': result['vader_compound']
                    },
                    'TextBlob': {
                        'polarity': result['tb_polarity'],
                        'subjectivity': result['tb_subjectivity']
                    },
                    'LR Model': result['lr_proba']
                })
        else:
            st.warning("Please paste a review first.")

# ── TAB 2: BULK ANALYSIS ────────────────────────────────────────
with tab2:
    st.markdown("### Upload a CSV of reviews")
    st.caption("CSV must have a column named `review_text`")

    # 💡 Added explicit double quotes around text lines to handle natural commas safely
    sample_csv_data = (
        'review_text\n'
        '"The item arrived broken and scratched right out of the box."\n'
        '"Absolutely incredible purchase! Highly recommend to everyone."\n'
        '"It was okay I guess, nothing special but works fine."\n'
        '"Worst mistake ever. Avoid this product entirely, total waste of money."\n'
        '"This product saved me so much time. Simply amazing features!"'
    )
    
    st.markdown(" Don't have a file ready? Download our test spreadsheet template below:")
    st.download_button(
        label="📥 Download Sample Test CSV File",
        data=sample_csv_data,
        file_name="sample_amazon_reviews.csv",
        mime="text/csv"
    )
    st.divider()

    uploaded = st.file_uploader("Upload CSV", type=['csv'])

    if uploaded:
        bulk_df = pd.read_csv(uploaded)
        st.write(f"Loaded {len(bulk_df):,} rows. Preview:")
        st.dataframe(bulk_df.head(3))

        if 'review_text' not in bulk_df.columns:
            st.error("CSV must contain a column named `review_text`")
        else:
            if st.button("Run Bulk Analysis", type="primary"):
                with st.spinner("Analyzing reviews..."):
                    bulk_df['cleaned'] = bulk_df['review_text'].apply(clean_text)
                    vader_results = bulk_df['review_text'].apply(
                        lambda x: sia.polarity_scores(str(x))['compound']
                    )
                    bulk_df['vader_compound'] = vader_results
                    bulk_df['vader_sentiment'] = bulk_df['vader_compound'].apply(sentiment_label)

                    tfidf_matrix = tfidf.transform(bulk_df['cleaned'])
                    bulk_df['ml_sentiment'] = lr.predict(tfidf_matrix)

                st.success("Done!")

                col1, col2 = st.columns(2)
                with col1:
                    dist = bulk_df['vader_sentiment'].value_counts().reset_index()
                    dist.columns = ['sentiment', 'count']
                    fig = px.pie(dist, values='count', names='sentiment',
                        title='VADER Sentiment Distribution',
                        color='sentiment',
                        color_discrete_map={'Positive':'#2ecc71','Negative':'#e74c3c','Neutral':'#f39c12'}
                    )
                    fig.update_layout(template='plotly_white')
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    dist2 = bulk_df['ml_sentiment'].value_counts().reset_index()
                    dist2.columns = ['sentiment', 'count']
                    fig2 = px.pie(dist2, values='count', names='sentiment',
                        title='ML Model Sentiment Distribution',
                        color='sentiment',
                        color_discrete_map={'Positive':'#2ecc71','Negative':'#e74c3c','Neutral':'#f39c12'}
                    )
                    fig2.update_layout(template='plotly_white')
                    st.plotly_chart(fig2, use_container_width=True)

                csv_out = bulk_df[['review_text','vader_sentiment','ml_sentiment','vader_compound']].to_csv(index=False)
                st.download_button("Download Results CSV", csv_out, "sentiment_results.csv", "text/csv")

st.divider()
st.caption("Built with VADER · TextBlob · Scikit-learn · Streamlit")
