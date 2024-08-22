from transformers import pipeline
from PIL import Image
import requests
import streamlit as st
from io import BytesIO

# Load BLIP model and processor
image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")

# Load sentiment analysis model
emotion_classifier = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

st.title("Image Description and Emotion Analysis")

st.write("Enter an image URL:")
image_url = st.text_input("Image URL")

if image_url:
  st.write("Image URL:", image_url)
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
else st.button("Generate Description and Analyze Emotion"):
  description = generate_description(image_url)
  st.write("Generated Description:", description)
  emotion = analyze_emotion(description)
  st.write("Emotional Tone:", emotion)
  # Display the image
  st.image(image_url, caption="Uploaded Image", use_column_width=True)
  # Display the generated description and emotional tone
  st.write("Generated Description:", description)
  st.write("Emotional Tone:", emotion)
  st.stop()
