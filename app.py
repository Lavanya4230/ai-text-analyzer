import streamlit as st
import fitz  # PyMuPDF
from summa.summarizer import summarize
from summa import keywords
from textblob import TextBlob
import spacy
import langdetect
from gtts import gTTS
import os
import difflib
from wordcloud import WordCloud
import nltk
from nltk.tokenize import sent_tokenize
import matplotlib.pyplot as plt
import wikipedia
from io import BytesIO
import base64

# Download dependencies
nltk.download('punkt')
nlp = spacy.load("en_core_web_sm")

# Helper functions
def extract_text_from_pdf(uploaded_file):
    text = ""
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    for page in doc:
        text += page.get_text("text") + "\n"
    doc.close()
    return text.strip() or "No text found in the PDF."

def caesar_cipher(text, shift):
    result = []
    for char in text:
        if char.isalpha():
            base = ord('A') if char.isupper() else ord('a')
            result.append(chr((ord(char) - base + shift) % 26 + base))
        else:
            result.append(char)
    return "".join(result)

def create_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color="white",
                          min_font_size=10, max_words=100).generate(text)
    return wordcloud

def save_audio(text):
    tts = gTTS(text=text, lang="en")
    output_path = "output.mp3"
    tts.save(output_path)
    return output_path

def audio_download_link(audio_path):
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    b64 = base64.b64encode(audio_bytes).decode()
    return f'<a href="data:audio/mp3;base64,{b64}" download="output.mp3">Download MP3</a>'

# Streamlit UI
st.title("🧠 AI Text Analyzer App")

st.sidebar.title("Upload and Choose Task")
uploaded_pdf = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
task = st.sidebar.selectbox("Select Task", [
    "Summarization", "Keyword Extraction", "Sentiment Analysis", "Named Entity Recognition",
    "Language Detection", "Text-to-Speech", "Grammar & Spell Checking", "Word Cloud",
    "Text Similarity Check", "Word Count Statistics", "Topic Description", "Encryption"
])

if uploaded_pdf:
    st.success("PDF uploaded successfully.")
    text = extract_text_from_pdf(uploaded_pdf)
    st.subheader("Extracted Text")
    st.text_area("Text", value=text, height=200)

    if task == "Summarization":
        ratio = st.slider("Summarization Ratio", 0.1, 1.0, 0.3)
        summary = summarize(text, ratio=ratio)
        st.subheader("Summary")
        st.write(summary or "Text too short to summarize.")

    elif task == "Keyword Extraction":
        extracted_keywords = keywords.keywords(text, scores=True)
        top_keywords = extracted_keywords[:10]
        st.subheader("Top Keywords")
        for kw in top_keywords:
            st.write(f"- {kw[0]} (Score: {round(kw[1], 3)})")

    elif task == "Sentiment Analysis":
        blob = TextBlob(text)
        sentiment = "Positive 😊" if blob.polarity > 0.1 else "Negative 😠" if blob.polarity < -0.1 else "Neutral 😐"
        st.write(f"*Sentiment*: {sentiment}")
        st.write(f"*Polarity*: {blob.polarity:.3f}")
        st.write(f"*Subjectivity*: {blob.subjectivity:.3f}")

    elif task == "Named Entity Recognition":
        doc = nlp(text)
        st.subheader("Named Entities")
        if doc.ents:
            for ent in doc.ents:
                st.write(f"{ent.text} ({ent.label_})")
        else:
            st.info("No named entities found.")

    elif task == "Language Detection":
        try:
            lang = langdetect.detect(text)
            confidence = langdetect.detect_langs(text)[0].prob
            st.write(f"*Language*: {lang.upper()}")
            st.write(f"*Confidence*: {round(confidence, 3)}")
        except:
            st.warning("Language detection failed.")

    elif task == "Text-to-Speech":
        if len(text) <= 5000:
            audio_path = save_audio(text)
            st.audio(audio_path, format="audio/mp3")
            st.markdown(audio_download_link(audio_path), unsafe_allow_html=True)
        else:
            st.warning("Text too long for TTS (max 5000 characters).")

    elif task == "Grammar & Spell Checking":
        blob = TextBlob(text)
        corrected = blob.correct()
        st.subheader("Corrected Text")
        st.write(str(corrected))

    elif task == "Word Cloud":
        try:
            wordcloud = create_wordcloud(text)
            st.image(wordcloud.to_array())
        except:
            st.warning("Text too short or invalid for word cloud.")

    elif task == "Text Similarity Check":
        text2 = st.text_area("Enter Second Text to Compare")
        if text2:
            similarity = difflib.SequenceMatcher(None, text, text2).ratio()
            st.write(f"*Similarity*: {round(similarity * 100, 2)}%")

    elif task == "Word Count Statistics":
        words = text.split()
        unique_words = len(set(words))
        sentences = sent_tokenize(text)
        avg_word_len = sum(len(w) for w in words) / len(words) if words else 0
        st.write(f"Total Words: {len(words)}")
        st.write(f"Unique Words: {unique_words}")
        st.write(f"Total Sentences: {len(sentences)}")
        st.write(f"Average Word Length: {round(avg_word_len, 2)}")

    elif task == "Topic Description":
        try:
            summary = wikipedia.summary(text, sentences=3)
            st.write(summary.replace('. ', '.\n'))
        except wikipedia.exceptions.DisambiguationError as e:
            st.warning(f"Ambiguous topic. Options include: {e.options}")
        except wikipedia.exceptions.PageError:
            st.warning("Topic not found on Wikipedia.")
        except Exception as e:
            st.error(f"Error: {str(e)}")

    elif task == "Encryption":
        shift = st.slider("Caesar Cipher Shift", 1, 25, 3)
        encrypted = caesar_cipher(text, shift)
        st.text_area("Encrypted Text", value=encrypted, height=150)
else:
    st.info("📄 Please upload a PDF file from the sidebar to get started.")