import streamlit as st
from transformers import pipeline
from PIL import Image
import requests
from io import BytesIO

# Load BLIP model and processor
image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")

# Load sentiment analysis model
emotion_classifier = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

def generate_description(image_url):
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))

    # Preprocess the image and generate description
    description = image_to_text(image)
    return description[0]['generated_text']

def analyze_emotion(description):
    # Analyze the emotional tone of the description
    result = emotion_classifier(description)
    return result[0]['label']

def main():
    st.title("AI-Powered Artwork Description and Emotional Tone Analysis")

    # Input: Image URL
    image_url = st.text_input("Enter the image URL:")

    if image_url:
        try:
            # Display the image
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Generate description
            description = generate_description(image_url)
            st.write("**Generated Description:**", description)

            # Analyze emotional tone
            emotion = analyze_emotion(description)
            st.write("**Emotional Tone:**", emotion)

        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
