import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import openai
import cv2
import torch

# Load the model
model_path = "trained_model (1).pt"
model = YOLO(model_path)


# Define your prediction function
def read_and_detect(image_array):
    image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    results = model.predict(image_rgb )
    result = results[0]

#     # Extract class names and probabilities
    class_names = result.names
    probs = result.probs.data.tolist()  # Convert probabilities to list
    max_prob_index = np.argmax(probs)   # Index of highest probability
    tumor_type = class_names[max_prob_index].upper()  # Get the class name with the highest probability
    confidence_score = probs[max_prob_index]
    return tumor_type,confidence_score


# Define your function to generate tumor information
def get_tumor_info(tumor_type):
    openai.api_key = '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'   # Replace with your API key

    tumor_descriptions = {
        0: "No Tumor detected.",
        1: "glioma",
        2: "pituitary tumor",
        3: "meningioma"
    }

    tumor_name = tumor_descriptions.get(tumor_type, "Unknown tumor type")

    prompt = f"Please provide detailed information about {tumor_name}. Include symptoms, treatment options, and general facts."

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message['content']


# Streamlit app
st.title("Brain Tumor Detection and Information Generator")

st.write("Kindly upload a brain scan image to predict if there is a tumor growing.")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Button to process the image
    if st.button('Process Image'):

        #Convert the image to RGB
        image_array = np.array(image)
        tumor_type,confidence_score = read_and_detect(image_array)
        tumor_info = get_tumor_info(tumor_type)
        st.write(f"Predicted Tumor Type: {tumor_type} and confidence score is{confidence_score} ")
        st.write("Information about the tumor:")
        st.write(tumor_info)

# former app

# import streamlit as st
# import torch
# from ultralytics import YOLO
# from PIL import Image
# import numpy as np
#
# # Load the model
# model_path = '/content/drive/MyDrive/trained_model (1).pt'
# model = YOLO(model_path)  # Adjust based on how you initialize your model
#
# # Streamlit app
# st.title("Brain Tumor Detection and Information Generator")
#
# st.write("Kindly upload a brain scan image to predict if there is a tumor growing.")
# # st.write("Do you want detailed information about the tumor ?")
#
#
# # Image upload
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
#
# if uploaded_file is not None:
#     image = Image.open(uploaded_file)
#     st.image(image, caption='Uploaded Image', use_column_width=True)
#
#     # Convert the image to a format suitable for your model
#     image_array = np.array(image)
#
#     # Predict tumor type
#     tumor_type = read_and_detect(image_array)
#
#     # Generate tumor information
#     tumor_info = get_tumor_info(tumor_type)
#
#     st.write(f"Predicted Tumor Type: {tumor_type}")
#     st.write("Information about the tumor:")
#     st.write(tumor_info)
#
#
