# Image Description and Emotion Analysis

## Description

This application uses a combination of image captioning and sentiment analysis models to generate a description of an image and analyze its emotional tone.

## Requirements

- Python 3.x
- Streamlit
- Transformers
- PIL
- Requests

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo

### Libraries and Models

1. **Libraries**:
   - `transformers`: Provides pre-trained models and pipelines for various NLP tasks.
   - `PIL` (Python Imaging Library): Used to handle image processing.
   - `requests`: Handles HTTP requests to fetch data from URLs.
   - `streamlit`: Framework for creating interactive web applications.
   - `io`: Provides tools for handling byte streams (in this case, for image data).

2. **Models**:
   - **BLIP** (`Salesforce/blip-image-captioning-large`): A model for generating captions from images.
   - **Sentiment Analysis** (`cardiffnlp/twitter-roberta-base-sentiment-latest`): A model for classifying the emotional tone of text.

### Streamlit App Workflow

1. **Title and Input**:
   - `st.title("Image Description and Emotion Analysis")`: Sets the title of the web application.
   - `st.write("Enter an image URL:")`: Provides instructions to the user.
   - `image_url = st.text_input("Image URL")`: Creates an input field where the user can enter the URL of an image.

2. **Processing the Image**:
   - **Image Fetching**:
     ```python
     response = requests.get(image_url)
     image = Image.open(BytesIO(response.content))
     ```
     - Fetches the image from the provided URL using `requests.get()`.
     - Converts the fetched image data into an `Image` object using `PIL.Image.open()`.

   - **Displaying the Image**:
     ```python
     st.image(image, caption="Uploaded Image", use_column_width=True)
     ```
     - Displays the image in the Streamlit app with a caption.

3. **Generating Description**:
   - `description = image_to_text(image)`: Uses the BLIP model to generate a description of the image.
   - `description_text = description[0]['generated_text']`: Extracts the generated text from the model output.

4. **Sentiment Analysis**:
   - `emotion = emotion_classifier(description_text)`: Analyzes the sentiment of the generated description using the sentiment analysis model.
   - `emotion_label = emotion[0]['label']`: Extracts the sentiment label from the model output.

5. **Displaying Results**:
   - `st.write("Generated Description:", description_text)`: Shows the generated image description.
   - `st.write("Emotional Tone:", emotion_label)`: Displays the emotional tone of the description.

### Summary

The app takes an image URL input from the user, fetches the image, generates a descriptive caption for the image, analyzes the sentiment of the caption, and displays both the caption and sentiment in the web application.
