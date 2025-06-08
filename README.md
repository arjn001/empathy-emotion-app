# Empathy Emotion Classifier 🧠💬

Real-time emotion detection from natural language — with **token-level SHAP explainability**.

🔗 **Live App** → [https://empathy-analysis.streamlit.app/](https://empathy-analysis.streamlit.app/)

---

### 👀 What It Does

Type anything.  
→ The app predicts the underlying **emotion**  
→ Then breaks down **which words influenced** that prediction the most.

- Built on 🤗 DistilBERT fine-tuned for emotional classification
- Interactive SHAP visualizations for **word-level impact**
- Simple, fast UI using Streamlit

---

### 🔍 Also Built: Emotion Lexicon (Full Dataset SHAP)

I analyzed the **entire EmpatheticDialogues dataset** using SHAP to generate:

- A CSV of top tokens per emotion (`emotion_lexicon_full.csv`)
- Average SHAP values per word, per emotion
- Coming soon: Lexicon Explorer (word clouds + emotion profiles)

This goes beyond single predictions — it's a **global look** at how AI maps language to emotions.

---

### ⚙️ Tech Stack

- `transformers` by Hugging Face  
- `torch` for model inference  
- `shap` for explainability  
- `streamlit` for UI & deployment

---

### 🧪 Run It Locally

```bash
git clone git@github.com:arjn001/empathy-emotion-app.git
cd empathy-emotion-app

pip install -r requirements.txt
streamlit run app.py
