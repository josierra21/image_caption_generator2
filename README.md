# Image Caption Generator 2

This project trains a CNN + LSTM image caption generator using VGG16 image features and the Flickr30k dataset.

## Project structure

```text
image_caption_generator2/
  image_captioner2.ipynb
  app.py
  requirements.txt
  artifacts/
  resource/
```

The Flickr30k dataset was pulled from Kaggle:

https://www.kaggle.com/datasets/eeshawn/flickr30k

## Setup

```bash
py -3.12 -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install ipykernel
python -m ipykernel install --user --name image-caption-vgg16-env --display-name "Python (.venv image caption VGG16)"
```

## Training

Open `image_captioner2.ipynb`, select the `.venv` kernel, and run the notebook from top to bottom.

Training creates these files in `artifacts/`:

```text
caption_model_vgg16.keras
caption_tokenizer_vgg16.pkl
caption_config_vgg16.pkl
vgg16_features.pkl
```

## Running the project

You can use the deployed app here:

https://joannas-image-caption-generator2.streamlit.app/

To run the app locally after training, use:

```bash
streamlit run app.py
```
