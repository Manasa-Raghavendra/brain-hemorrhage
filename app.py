import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, render_template, request, redirect, flash, url_for
import numpy as np

app = Flask(__name__)
app.secret_key = 'brain_hemorrhage_detection'

# Load the trained model
MODEL_PATH = "resnet18_brain_ct_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model architecture (ResNet18)
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 1)

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading the model: {e}")

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(image_path):
    """
    Predicts if the uploaded image shows signs of brain hemorrhage.
    """
    try:
        image = Image.open(image_path).convert("RGB")
        image = transform(image)
        image = image.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image).item()
            prediction = "Hemorrhage" if output > 0.5 else "Normal"
            probability = torch.sigmoid(torch.tensor(output)).item()

        return prediction, probability
    except Exception as e:
        print(f"Error in prediction: {e}")
        return "Error", 0.0

@app.route('/')
def home():
    """
    Renders the homepage.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles file upload and prediction requests.
    """
    try:
        if 'file' not in request.files:
            flash('No file uploaded')
            return redirect(url_for('home'))

        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(url_for('home'))

        if file:
            # Save the uploaded file
            upload_folder = 'static/uploads'
            if not os.path.exists(upload_folder):
                os.makedirs(upload_folder)

            file_path = os.path.join(upload_folder, file.filename)
            file.save(file_path)

            # Perform prediction
            prediction, probability = predict_image(file_path)

            # Check if prediction was successful
            if prediction == "Error":
                flash('An error occurred during prediction')
                return redirect(url_for('home'))

            # Render the result
            return render_template('index.html', prediction=prediction, probability=probability, image_path=file_path)
    except Exception as e:
        print(f"Error in /predict route: {e}")
        flash('An unexpected error occurred')
        return redirect(url_for('home'))
    
#updated
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
