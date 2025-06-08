import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components



# Title
st.set_page_config(page_title="Empathy Emotion Classifier")
st.title("🧠 Empathy Emotion Classifier")
st.markdown("Enter a sentence and discover the underlying emotion.")

# Check device
device = 0 if torch.cuda.is_available() else -1

# Load model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "bhadresh-savani/distilbert-base-uncased-emotion"
    )
    clf = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)
    return clf, tokenizer

classifier, tokenizer = load_model_and_tokenizer()


# Input box
text = st.text_input("💬 Your sentence:", "I’m really sorry you’re feeling that way.")

# Predict and show results
if text:
    with st.spinner("Analyzing emotion..."):
        result = classifier(text)[0]
        st.success(f"Predicted Emotion: **{result['label']}**")
        st.caption(f"Confidence: {result['score']:.2f}")

        # SHAP explanation
       
        try:
            st.subheader("🔍 Why this prediction?")
            explainer = shap.Explainer(lambda x: [classifier(t)[0]['score'] for t in x], tokenizer)
            shap_values = explainer([text])
            html = shap.plots.text(shap_values[0], display=False)
            components.html(html, height=300)
        except Exception as e:
            st.warning("SHAP explanation failed. Try a simpler sentence or refresh.")
            st.text(str(e))