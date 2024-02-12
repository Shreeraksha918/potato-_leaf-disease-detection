from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load your pre-trained model
model = load_model('potatoes.h5')

def process_image(image):
    # Preprocess the image (resize, normalize, etc.)
    # Implement the preprocessing steps specific to your model
    # For example, if your model expects images of size (224, 224) and normalized between 0 and 1:
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def detect_disease(image):
    processed_image = process_image(image)
    # Use your pre-trained model to predict the disease
    # Replace this with your actual prediction code
    prediction = model.predict(processed_image)
    # For simplicity, let's assume your model outputs a label index
    # You should map this index to the actual disease name
    # For example:
    disease_labels = ['Healthy', 'Disease1', 'Disease2']  # Replace with actual disease names
    predicted_label_index = np.argmax(prediction)
    predicted_disease = disease_labels[predicted_label_index]
    return predicted_disease

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Get the uploaded file
        file = request.files['file']
        if file:
            # Read the image
            img = Image.open(io.BytesIO(file.read()))
            # Detect disease
            result = detect_disease(img)
            return render_template('result.html', result=result)
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
