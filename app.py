import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Amazon Review Sentiment Analyzer",
    page_icon="🛍️",
    layout="centered"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "martigol/classification-distilbert"

LABELS = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME
    )

    model.eval()

    return tokenizer, model


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_sentiment(text, tokenizer, model):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True
    )

    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()

    sentiment = LABELS[predicted_class]

    confidence = probabilities[0][predicted_class].item()

    return sentiment, confidence, probabilities[0]


# ============================================================
# TITLE
# ============================================================

st.title("🛍️ Amazon Review Sentiment Analyzer")

st.write(
    "Analyze the sentiment of an Amazon customer review "
    "using a fine-tuned DistilBERT model."
)


# ============================================================
# LOAD MODEL
# ============================================================

with st.spinner("Loading AI model..."):

    tokenizer, model = load_model()


st.success("Model loaded successfully!")


# ============================================================
# USER INPUT
# ============================================================

review = st.text_area(
    "Enter an Amazon review:",
    placeholder="Example: This product is amazing. "
                "I absolutely love it!",
    height=180
)


# ============================================================
# ANALYZE BUTTON
# ============================================================

if st.button("🔍 Analyze Sentiment", use_container_width=True):

    if not review.strip():

        st.warning("Please enter a review first.")

    else:

        sentiment, confidence, probabilities = predict_sentiment(
            review,
            tokenizer,
            model
        )

        st.divider()

        st.subheader("Sentiment")

        # Display main prediction

        if sentiment == "Positive":

            st.success(
                f"😊 {sentiment}"
            )

        elif sentiment == "Negative":

            st.error(
                f"😞 {sentiment}"
            )

        else:

            st.info(
                f"😐 {sentiment}"
            )


        # Confidence

        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%"
        )


        # ====================================================
        # PROBABILITIES
        # ====================================================

        st.subheader("Prediction probabilities")

        negative_probability = probabilities[0].item()
        neutral_probability = probabilities[1].item()
        positive_probability = probabilities[2].item()

        st.write(
            f"🔴 Negative: "
            f"{negative_probability * 100:.2f}%"
        )

        st.progress(
            float(negative_probability)
        )

        st.write(
            f"🟡 Neutral: "
            f"{neutral_probability * 100:.2f}%"
        )

        st.progress(
            float(neutral_probability)
        )

        st.write(
            f"🟢 Positive: "
            f"{positive_probability * 100:.2f}%"
        )

        st.progress(
            float(positive_probability)
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Powered by DistilBERT · Amazon Customer Reviews "
    "Sentiment Analysis"
)