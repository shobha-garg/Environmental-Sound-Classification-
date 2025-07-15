from flask import Flask, render_template, request, jsonify, url_for
import os
import librosa
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle
import uuid
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model
try:
    model = tf.keras.models.load_model('saved_models/audio_classification.hdf5')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Load or create label encoder
try:
    # Try to load saved label encoder first
    with open('saved_models/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    print("Label encoder loaded from file!")
except:
    # Fallback: Create label encoder with UrbanSound8K classes
    # UPDATE THESE CLASS NAMES TO MATCH YOUR ACTUAL MODEL'S CLASSES
    class_names = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 
                   'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 
                   'siren', 'street_music']
    
    label_encoder = LabelEncoder()
    label_encoder.fit(class_names)
    print("Label encoder created with default UrbanSound8K classes")
    print("Note: Update class_names in app.py to match your model's actual classes")

def mfccs(data, sample_rate):
    """Extract MFCC features from audio data"""
    mfccs = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=13)
    mfccs = np.mean(mfccs.T, axis=0)
    return mfccs

def features_extractor(file_path):
    """Extract features from audio file"""
    try:
        data, sample_rate = librosa.load(file_path)
        mfccs_feature = mfccs(data, sample_rate)
        return mfccs_feature
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def predict_audio_class(file_path):
    """Predict the class of an audio file"""
    if model is None:
        return "Model not loaded"
    
    try:
        # Extract features
        features = features_extractor(file_path)
        if features is None:
            return "Error extracting features"
        
        # Reshape for prediction
        prediction_feature = features.reshape(1, -1)
        
        # Make prediction
        prediction_probs = model.predict(prediction_feature, verbose=0)
        prediction = np.argmax(prediction_probs, axis=1)
        
        # Get class name
        predicted_class = label_encoder.inverse_transform(prediction)[0]
        
        return predicted_class
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Error during prediction"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        print("Upload request received")
        
        if 'audio_file' not in request.files:
            print("No audio_file in request")
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['audio_file']
        print(f"File received: {file.filename}")
        
        if file.filename == '':
            print("Empty filename")
            return jsonify({'error': 'No file selected'}), 400
        
        if file and file.filename.lower().endswith('.wav'):
            # Generate unique filename
            unique_filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            print(f"Saving file to: {file_path}")
            
            # Save file
            file.save(file_path)
            print("File saved successfully")
            
            # Check if model is loaded
            if model is None:
                print("Model is not loaded")
                return jsonify({'error': 'Model not loaded. Please check saved_models/audio_classification.hdf5'}), 500
            
            # Make prediction
            print("Making prediction...")
            predicted_class = predict_audio_class(file_path)
            print(f"Prediction: {predicted_class}")
            
            # Check for prediction errors
            if predicted_class.startswith("Error") or predicted_class == "Model not loaded":
                return jsonify({'error': predicted_class}), 500
            
            return jsonify({
                'success': True,
                'predicted_class': predicted_class,
                'audio_url': url_for('uploaded_file', filename=unique_filename)
            })
        
        else:
            print("Invalid file type")
            return jsonify({'error': 'Please upload a .wav file'}), 400
            
    except Exception as e:
        print(f"Unexpected error in upload_file: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    from flask import send_from_directory
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)