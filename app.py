import os
import pickle
import warnings
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

PROJECT_DIR = Path(__file__).parent
ARTIFACT_DIR = PROJECT_DIR / "artifacts"
MODEL_FILE = ARTIFACT_DIR / "caption_model_vgg16.keras"
TOKENIZER_FILE = ARTIFACT_DIR / "caption_tokenizer_vgg16.pkl"
CONFIG_FILE = ARTIFACT_DIR / "caption_config_vgg16.pkl"

st.set_page_config(page_title="Image Caption Generator 2", layout="centered")
st.title("Image Caption Generator 2")
st.write(
    "Upload a photo and the app will try to describe what it sees. "
    "This is a simple CNN + LSTM model using VGG16 and tested with the Flickr30k image caption dataset, "
    "so it works best with common everyday images and may describe unusual objects or animals in a more general way."
)


def artifact_exists() -> bool:
    return MODEL_FILE.exists() and TOKENIZER_FILE.exists() and CONFIG_FILE.exists()


@st.cache_resource
def load_caption_assets():
    caption_model = load_model(MODEL_FILE)
    with open(TOKENIZER_FILE, "rb") as file:
        tokenizer = pickle.load(file)
    with open(CONFIG_FILE, "rb") as file:
        config = pickle.load(file)

    base_model = VGG16(weights="imagenet")
    image_encoder = Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output)
    return caption_model, tokenizer, config, image_encoder


def word_from_id(tokenizer, token_id):
    return tokenizer.index_word.get(token_id)


def extract_image_feature(image: Image.Image, image_encoder):
    image = image.convert("RGB").resize((224, 224))
    image_array = np.asarray(image, dtype="float32")
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)
    return image_encoder.predict(image_array, verbose=0)[0]


def generate_caption(model, tokenizer, feature_vector, max_caption_length):
    words = ["startseq"]
    for _ in range(max_caption_length):
        token_ids = tokenizer.texts_to_sequences([" ".join(words)])[0]
        token_ids = pad_sequences([token_ids], maxlen=max_caption_length, padding="post")
        probabilities = model.predict(
            {"image_features": np.asarray([feature_vector]), "caption_tokens": token_ids},
            verbose=0,
        )
        predicted_id = int(np.argmax(probabilities))
        predicted_word = word_from_id(tokenizer, predicted_id)
        if predicted_word is None:
            break
        words.append(predicted_word)
        if predicted_word == "endseq":
            break
    return " ".join(words).replace("startseq", "").replace("endseq", "").strip()


if not artifact_exists():
    st.warning(
        "Train the model in image_captioner2.ipynb first. The app needs caption_model_vgg16.keras, "
        "caption_tokenizer_vgg16.pkl, and caption_config_vgg16.pkl inside the artifacts folder."
    )
    st.stop()

caption_model, caption_tokenizer, caption_config, vgg16_encoder = load_caption_assets()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    uploaded_image = Image.open(uploaded_file)
    st.image(uploaded_image, caption="Uploaded image", use_container_width=True)

    if st.button("Generate caption"):
        with st.spinner("Generating caption..."):
            feature_vector = extract_image_feature(uploaded_image, vgg16_encoder)
            caption = generate_caption(
                caption_model,
                caption_tokenizer,
                feature_vector,
                caption_config["max_caption_length"],
            )
        st.subheader("Generated caption")
        st.write(caption if caption else "No caption generated. Try another image or train longer.")
else:
    st.info("Upload a JPG or PNG image to generate a caption.")
