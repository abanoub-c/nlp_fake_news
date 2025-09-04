import streamlit as st
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# ----------------------
# Load Models
# ----------------------
@st.cache_resource
def load_models():
    # Load Fake News model
    fake_model = DistilBertForSequenceClassification.from_pretrained("saved_model2")
    fake_tokenizer = DistilBertTokenizerFast.from_pretrained("saved_model2")

    # Load Sentiment model
    sent_model = DistilBertForSequenceClassification.from_pretrained("sentiment_model")
    sent_tokenizer = DistilBertTokenizerFast.from_pretrained("sentiment_model")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fake_model.to(device).eval()
    sent_model.to(device).eval()

    return fake_model, fake_tokenizer, sent_model, sent_tokenizer, device


fake_model, fake_tokenizer, sent_model, sent_tokenizer, device = load_models()

# ----------------------
# Prediction functions
# ----------------------
def predict_fake_news(text):
    inputs = fake_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = fake_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).cpu().item()
        confidence = probs[0, pred_class].cpu().item()
    label = "🟢 Real News" if pred_class == 1 else "🔴 Fake News"
    return label, confidence


def predict_sentiment(text):
    inputs = sent_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = sent_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).cpu().item()
        confidence = probs[0, pred_class].cpu().item()

    # Assuming labels: 0 = Negative, 1 = Neutral, 2 = Positive
    labels = ["😡 Negative", "😐 Neutral", "😊 Positive"]
    return labels[pred_class], confidence


# ----------------------
# Streamlit UI
# ----------------------
st.set_page_config(page_title="Fake News & Sentiment Analyzer", page_icon="📰", layout="centered")
st.title("📰 Fake News & Sentiment Analyzer")
st.markdown("Enter a news headline or short article, and the models will classify it as **Fake/Real** and also show **Sentiment Analysis**.")

user_input = st.text_area("✍️ Paste your text here:", height=150)

if st.button("🔍 Analyze"):
    if user_input.strip():
        # Fake news prediction
        fake_label, fake_conf = predict_fake_news(user_input)
        st.subheader(f"📰 Fake News Detection: {fake_label}")
        st.write(f"Confidence: **{fake_conf:.2%}**")
        st.progress(int(fake_conf * 100))

        # Sentiment prediction
        sent_label, sent_conf = predict_sentiment(user_input)
        st.subheader(f"💬 Sentiment Analysis: {sent_label}")
        st.write(f"Confidence: **{sent_conf:.2%}**")
        st.progress(int(sent_conf * 100))
    else:
        st.warning("⚠️ Please enter some text to analyze.")
