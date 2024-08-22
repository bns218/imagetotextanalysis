%%writefile app.py
from transformers import pipeline
from PIL import Image
import requests
import streamlit as st
from io import BytesIO

# Load BLIP model for image captioning
image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")

# Load sentiment analysis model
emotion_classifier = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

st.title("Image Description and Emotion Analysis")

st.write("Enter an image URL:")
image_url = st.text_input("Image URL")

if image_url:
    # Fetch the image from the URL
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    
    # Display the image in the Streamlit app
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Generate description from the image
    description = image_to_text(image)
    description_text = description[0]['generated_text']
    
    # Perform sentiment analysis on the generated description
    emotion = emotion_classifier(description_text)
    emotion_label = emotion[0]['label']
    
    # Display results
    st.write("Generated Description:", description_text)
    st.write("Emotional Tone:", emotion_label)
