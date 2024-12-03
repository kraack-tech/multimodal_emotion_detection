# Import libraries
import pandas as pd
import numpy as np
import librosa
import pickle
from tensorflow.keras.models import load_model
#import os


# Load the scaler from the file
with open('models/voice/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Load the encoder from the file
with open('models/voice/encoder.pkl', 'rb') as file:
    encoder = pickle.load(file)

model = load_model('models/voice/emotion_model_CSV.h5')


# Noise: Add randomised noise to audio signal
def add_noise(data):
    # Generate noise and add it to the input audio signal
    amp = 0.025 * np.random.uniform() * np.amax(data) # Generate noise
    return data + amp * np.random.normal(size=data.shape) # Apply to the input audio signal

# Dynamic compression: Apply dynamic compression to reduce volume spikes
def dynamic_compression(data, threshold=0.1, ratio=1.5):
    mask = np.abs(data) > threshold # Create mask using absolute signal value and threshold parameter 
    # Apply compression to values above the threshold
    compressed_data = np.where(mask, 
                               threshold + (data - threshold) / ratio, 
                               data)  
    return compressed_data

# Pitch shift: Shift the pitch of the input audio signal
def pitch_shift(data, sampling_rate):
    pitch_factor = np.random.uniform(-0.5, 0.5)  # Randomize pitch shift (-0.5 to 0.5)
    return librosa.effects.pitch_shift(y=data, sr=sampling_rate, n_steps=pitch_factor) # Apply to the input audio signal


# Modified audio features extraction function from the  'extract_audio_features' notebook.
def extract_audio_features(data, sr=22050, frame_length=2048, hop_length=512):
    augmentations = [
        ("Original", lambda x: x),  # Original audio
        ("Noise", add_noise),  # Noise augmentation
        ("Dynamic Compression", lambda x: dynamic_compression(x)),  # Dynamic compression augmentation
        ("Dynamic Compression and Noise", lambda x: add_noise(dynamic_compression(x))),  # Dynamic compression + noise augmentation
        ("Pitch Shift", lambda x: pitch_shift(x, sr)),  # Pitch shift augmentation
        ("Pitch Shift and Noise", lambda x: add_noise(pitch_shift(x, sr))),  # Pitch shift + noise augmentation
    ]

    # Initialize list for extracted audio features
    audio_features = [] 

    # Iterate over augmentations to apply and extract audio features
    for name, augmentation in augmentations:
        try:
            augmented_data = augmentation(data)
            
            # Extract features (ZCR, RMS, MFCC)
            features = np.hstack((
                np.squeeze(librosa.feature.zero_crossing_rate(augmented_data, frame_length=frame_length, hop_length=hop_length)),
                np.squeeze(librosa.feature.rms(y=augmented_data, frame_length=frame_length, hop_length=hop_length)),
                np.squeeze(np.ravel(librosa.feature.mfcc(y=augmented_data, sr=sr, n_fft=frame_length, hop_length=hop_length).T))
            ))

            # Ensure the feature vector has the expected length (2684)
            if features.size < 2684:
                #print(f"Warning: Feature size is smaller than expected for {name}, padding with zeros.")
                # Pad with zeros if the number of features is less than 2684
                features = np.pad(features, (0, 2684 - features.size), mode='constant')
            audio_features.append(features)
            
        except Exception as e:
            print(f"Error processing {name} audio: {e}")
            continue
            
    return np.array(audio_features[-1])


# Loads audio, extracts features, reshapes, and makes predictions on the input audio file.
def voice_prediction(path):
    # Load the audio file
    # Load the audio file
    audio_data, sample_rate = librosa.load(path, duration=2.81, offset=0.39)
    
    # Extract audio features
    features = extract_audio_features(audio_data)
        
    # Reshape and scale the extracted features
    features = np.reshape(features, newshape=(1, -1))  # Flatten features
    scaled_features = scaler.transform(features)       # Scaling
    final_features = np.expand_dims(scaled_features, axis=2)  # Add dimension

    # Make prediction using the model
    predictions = model.predict(final_features, verbose=0)
    
    # Get the class labels from the encoder
    labels = encoder.categories_[0]

    # Calculate probabilities and normalize them to percentages
    probabilities = {label: f"{prob * 100:.2f}" for label, prob in zip(labels, predictions[0])}

    # Inverse transform to get the most probable emotion
    predicted_emotion = encoder.inverse_transform(predictions)

    # Add the predicted emotion to the result dictionary
    probabilities['predicted_emotion'] = predicted_emotion[0][0]

    return probabilities




