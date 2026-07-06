# Environmental Sound Classification

An end-to-end deep learning application for classifying environmental sounds using Convolutional Neural Networks (CNNs). The project identifies sounds such as dog bark, drilling, engine idling, sirens, and other urban environmental noises.

---

## Overview

This project was developed to demonstrate the complete workflow of an audio classification system, including data preprocessing, feature extraction, model training, evaluation, and deployment through a simple web interface.

The model uses Mel Frequency Cepstral Coefficients (MFCCs) extracted from audio signals to train a deep learning model capable of recognizing multiple environmental sound classes.

---

## Features

- Environmental sound classification
- MFCC-based audio feature extraction
- CNN-based deep learning model
- Interactive Flask web application
- Audio file upload and prediction
- Pre-trained model included

---

## Technologies Used

- Python
- TensorFlow / Keras
- Librosa
- NumPy
- Flask
- HTML
- CSS

---

## Dataset

This project uses the **UrbanSound8K** dataset containing 8,732 labeled urban sound clips across multiple environmental sound categories.

Example classes include:

- Dog Bark
- Drilling
- Engine Idling
- Siren
- Air Conditioner
- Car Horn
- Children Playing
- Street Music

---

## Workflow

1. Load audio file
2. Extract MFCC features
3. Preprocess audio
4. Perform model inference
5. Predict environmental sound class
6. Display prediction through the web interface

---

## Results

The trained model achieves approximately **70% classification accuracy** on the evaluation dataset and can classify multiple urban environmental sounds through an interactive web application.

---

## Future Improvements

- Increase classification accuracy using transfer learning
- Add real-time microphone input
- Support additional environmental sound classes
- Deploy using Docker or cloud services

---

## Note

The repository currently contains the trained model, notebook, and Flask application used for experimentation and demonstration purposes.
