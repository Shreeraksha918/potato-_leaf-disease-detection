import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import io
from flask import Flask, render_template, request,jsonify

app = Flask(__name__)

# Load your pre-trained model
model = load_model('potatoes.h5')

def process_image(image):
    # Resize the image to match the expected input shape of your model
    image = image.resize((256, 256))
    # Convert the image to a numpy array
    image = img_to_array(image)
    # Normalize the pixel values to the range [0, 1]
    image = image / 255.0
    # Add a batch dimension to the image
    image = np.expand_dims(image, axis=0)
    return image

def detect_disease(image):
    processed_image = process_image(image)
    # Use your pre-trained model to predict the disease
    prediction = model.predict(processed_image)
    # For simplicity, let's assume your model outputs a label index
    # You should map this index to the actual disease name
    # For example:
    disease_labels = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']  # Replace with actual disease names
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
            return jsonify({'result': result})  # Return result as JSON
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
