from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
import pickle
import os
import cv2

app = Flask(__name__)

# Load all models dynamically from the `trained_models` directory
model_directory = 'trained_models'
models = {}
for filename in os.listdir(model_directory):
    if filename.endswith('.pkl'):
        model_name = filename.split('.')[0]
        models[model_name] = pickle.load(open(os.path.join(model_directory, filename), 'rb'))

# Function to preprocess a single image
def preprocess_image(image, img_size=(64, 64)):
    if image is not None:
        image = cv2.resize(image, img_size)
        return image
    else:
        print(f"Error: Unable to load image.")
        return None

# Feature extraction function
def extract_histogram_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / np.sum(hist)
    return hist

def extract_hog_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    winSize = (64, 64)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    features = hog.compute(gray)
    return features.flatten()

def extract_color_moments(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    moments = []
    for channel in range(3):
        channel_data = hsv[:,:,channel]
        mean = np.mean(channel_data)
        std = np.std(channel_data)
        skewness = np.mean(((channel_data - mean) / std) ** 3)
        moments.extend([mean, std, skewness])
    return np.array(moments)

def extract_edge_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_count = np.sum(edges > 0)
    edge_density = edge_count / (edges.shape[0] * edges.shape[1])
    edge_mean = np.mean(edges[edges > 0])
    edge_std = np.std(edges[edges > 0])
    return np.array([edge_count, edge_density, edge_mean, edge_std])

def extract_features(image, img_size=(64, 64)):
    features = []
    image = preprocess_image(image, img_size)
    if image is not None:
        features.extend(extract_histogram_features(image))
        features.extend(extract_hog_features(image))
        features.extend(extract_color_moments(image))
        features.extend(extract_edge_features(image))
        return np.array(features)
    return None

# Function to preprocess the image for prediction
def preprocess_image_for_prediction(image):
    features = extract_features(image)
    if features is not None:
        features = features.reshape(1, -1)  # Ensure the features are in the right shape
        return features
    return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    label_encoder = {"Lung Adenocarcinoma": 0, "lung tissue": 1, "Lung Squamous Cell Carcinoma": 2}
    label_decoder = {v: k for k, v in label_encoder.items()}

    # Symptoms and treatment dictionary
    condition_info = {
        "lung_n": {
            "symptoms": "No symptoms, normal lung tissue.",
            "treatment": "No treatment required as this is a normal condition."
        },
        "lung_scc": {
            "symptoms": "Persistent cough, chest pain, shortness of breath, blood in sputum, weight loss.",
            "treatment": "Chemotherapy, radiation therapy, targeted therapy, and surgery (depending on stage)."
        },
        "lung_aca": {
            "symptoms": "Coughing, difficulty breathing, chest pain, fatigue, weight loss.",
            "treatment": "Surgery, chemotherapy, radiation therapy, and targeted therapies based on genetic mutations."
        }
    }

    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    try:
        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'error': 'Unable to decode image'}), 400

        processed_image = preprocess_image_for_prediction(image)
        if processed_image is None:
            return jsonify({'error': 'Error processing the image'}), 400

        # Predict using all models and average the scores
        model_predictions = []
        for model_name, model in models.items():
            prediction = model.predict_proba(processed_image)[0]  # Get probability scores
            model_predictions.append(prediction)

        avg_prediction = np.mean(model_predictions, axis=0)  # Average probabilities of class from all models
        final_prediction = np.argmax(avg_prediction)  # Class with highest average probability

        predicted_label = label_decoder.get(final_prediction, "Unknown")

        # Get symptoms and treatment for the predicted label
        condition = "lung_n" if predicted_label == "lung tissue" else ("lung_scc" if predicted_label == "Lung Squamous Cell Carcinoma" else "lung_aca")
        symptoms = condition_info.get(condition, {}).get("symptoms", "No data available")
        treatment = condition_info.get(condition, {}).get("treatment", "Consult a doctor for more details")

        return jsonify({
            'prediction': predicted_label,
            'symptoms': symptoms,
            'treatment': treatment,
            'average_scores': avg_prediction.tolist()
        })
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        return jsonify({'error': f"Error during prediction: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
