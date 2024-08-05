*<h1>README for Brain Tumor Detection App* 

<h2>Overview</h2>
Our brain tumor detection application app is a web application that uses a deep learning model, YoloV8, to analyze uploaded medical brain images and determine whether they show a brain tumor or not. If a tumor is detected, the app would then classify the tumor under the 3 main classes of brain tumors and provide detailed information to the user about the tumor type. We built the app with Flask and used the YoloV8 model for our image classification and the ChatGPT API for generating descriptive information.

<h2>How to Run the Code</h2>
- Download the datasets and the pretrained model (.pt file) <br>
- Change the directory of the training dataset and the pretrained model to suit your current working directory <br>
- Run the code
- The model creates a folder called "Runs" in your drive and in it, the best trained model and the confusion matrix are stored.

<h2>How to Deploy The App</h2>
- Download the app.py file <br>
- Change directory of pretrained model to suit your directory and run <br>
- Click on the link <br>

<h2>How To Use The App</h2>

1. *Upload an Image:* Use the file upload option to select and upload a brain scan image from your device. Supported image formats include .jpg, .jpeg and .png.

2. Once the image is uploaded, click on the Process image button, the app will then analyze the image, display the results, that is, whether a tumor is present and if so the type of brain tumor present, and display some detailed information about the tumor.

<h2>Requirements</h2>
- Python version 3.12.4 <br>
- Flask   <br>
- Ultralytics (for YoloV8)  <br>

<h2>Technical Details</h2>
Model : YOLOv8 for brain tumor detection and classification. <br>
API : OpenAI's GPT-3.5-turbo for generating descriptive information about tumor types. <br>
Framework : Flask for the web interface. <br>

<h2>Troubleshooting</h2>
1. Issue: App fails to load images. <br>
Solution: Ensure the image format is either .jpg, .jpeg or.png and the file is not corrupted. <br>2. Issue: The app crashes or returns an error.
Solution: Check the console for error messages and ensure all dependencies are correctly installed
