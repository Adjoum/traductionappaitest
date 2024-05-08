import streamlit as st
from audio_recorder_streamlit import audio_recorder
from faster_whisper import WhisperModel
import os
from groq_translation import groq_translate
from gtts import gTTS

# Set page config
st.set_page_config(page_title='Adjoumani IA', page_icon='🎤')

# Set page title
st.title("Polyglotte instantané : L'IA qui te fait parler toutes les langues")

languages = {
    "Portuguais": "pt",
    "Anglais": "en",
    "Espagnol": "es",
    "Allemand": "de",
    "Français": "fr",
    "Italien": "it",
    "Néerlandais": "nl",
    "Russe": "ru",
    "Japonnais": "ja",
    "Chinois": "zh",
    "Koréen": "ko"
}

# Language selection
option = st.selectbox(
    "Selectionnez la langue dans laquelle vous voulez la traduction:",
    languages,
    index=None,
    placeholder="Selectionnez une langue...",
)

# Load whisper model
model = WhisperModel("base", device="cpu", compute_type="int8", cpu_threads=int(os.cpu_count() / 2))


# Speech to text
def speech_to_text(audio_chunk):
    segments, info = model.transcribe(audio_chunk, beam_size=5)
    speech_text = " ".join([segment.text for segment in segments])
    return speech_text


# Text to speech
def text_to_speech(translated_text, language):
    file_name = "speech.mp3"
    my_obj = gTTS(text=translated_text, lang=language)
    my_obj.save(file_name)
    return file_name


# Record audio
audio_bytes = audio_recorder()
if audio_bytes and option:
    # Display audio player
    st.audio(audio_bytes, format="audio/wav")

    # Save audio to file
    with open('audio.wav', mode='wb') as f:
        f.write(audio_bytes)

    # Speech to text
    st.divider()
    with st.spinner('Transcription en cours...'):
        text = speech_to_text('audio.wav')
    st.subheader('Parole transcrite en texte')
    st.write(text)

    # Groq translation
    st.divider()
    with st.spinner('Traduction en cours. Veillez patienter un instant SVP...'):
        translation = groq_translate(text, 'en', option)
    st.subheader('Parole traduite en ' + option)
    st.write(translation.text)

    # Text to speech
    audio_file = text_to_speech(translation.text, languages[option])
    st.audio(audio_file, format="audio/mp3")