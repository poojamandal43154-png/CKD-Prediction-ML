import streamlit as st
from deep_translator import GoogleTranslator


LANGUAGES = {
    "English": "en",
    "Telugu": "te",
    "Hindi": "hi"
}


@st.cache_data
def translate_text(text, language):

    if language == "English":
        return text

    return GoogleTranslator(
        source="en",
        target=LANGUAGES[language]
    ).translate(text)


def T(text, language):
    return translate_text(text, language)