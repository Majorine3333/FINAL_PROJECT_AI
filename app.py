from flask import Flask, request, render_template, jsonify
from ultralytics import YOLO
from PIL import Image
import numpy as np
import openai
import torch
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access environment variables
api_key = os.getenv('OPENAI_API_KEY')

# Load the model
model_path = "C:/Users/hp/Downloads/trained_model (2).pt"
model = YOLO(model_path)

app = Flask(__name__)

# Define the target size for the YOLO model
TARGET_SIZE = (640, 640)  # Resize to this size

def resize_image(image, target_size=TARGET_SIZE):
    # Resize image to the target size
    if image.mode != 'RGB':
        image = image.convert('RGB')
    # Resize image to the target size
    resized_image = image.resize(target_size, Image.BILINEAR)
    return np.array(resized_image)

# Define your prediction function
def read_and_detect(image_array):
    # Resize image to match model input requirements
    image_array_resized = resize_image(Image.fromarray(image_array))
    
    # Convert to tensor and adjust dimensions for the model
    image_tensor = torch.from_numpy(image_array_resized).float()
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # Shape (1, 3, 640, 640)
    
    # Perform detection
  
    results = model.predict(image_tensor)
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
    openai.api_key = api_key

    tumor_descriptions = {
        0: "notumor",
        1: "glioma",
        2: "pituitary",
        3: "meningioma"
    }

    tumor_name = tumor_descriptions.get(tumor_type, "Unknown tumor type")

    prompt = f"Please provide detailed information about {tumor_name}. Include symptoms, treatment options, and general facts."

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message['content']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"error": "No file part"})
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"})
        if file:
            image = Image.open(file)
            image_array = np.array(image)
            tumor_type,confidence_score = read_and_detect(image_array)
            tumor_info = get_tumor_info(tumor_type)
            return render_template('result.html', tumor_type=tumor_type,confidence_score=confidence_score, tumor_info=tumor_info)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

