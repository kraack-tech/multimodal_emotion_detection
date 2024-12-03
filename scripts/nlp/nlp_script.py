import numpy as np
import re
import pickle
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch
import whisper

# Ignore warning (Whisper FP16)
import warnings
warnings.filterwarnings("ignore")

# Preprocesses text strings
def preprocess_text(text, stop_words):
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove non-alphabetic characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove input stopwords
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    
    return ' '.join(filtered_words)


# Load the tokenizer pickle file (created with the 'nlp_model.ipynb' notebook)
with open('models/nlp/tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

# Define stopwords
nltk_stop_words = set(stopwords.words('english')) # NLTK stopwords
custom_stop_words = { # Custom stopwords 
    "im", 'days', 'well', 'quite', 'look', 'find', 'come', 'year','lot', 'part', 'take',
    'href', 'every', 'able', 'though','left', 'need', 'new', 'http', 'sure', 'around', 'say',
    'also', 'work', 'today', 'pretty', 'feels', 'going', 'feelings', 'back', 'way', 'always',
    'things', "feel", "thats", "one", 'actually', 'right', 'many', 'thing', 'seen', 'thought',
    'believe', 'didnt', 'want', 'time', 'makes', "even", "day", "go", "made", "yeah", "man", 
    'youre', 'ive', 'much', 'good', "know", 'see', 'cant', 'never', "got", 'think', "would",
    "still", "dont", "people", "like", "really", "get", "name", "i", "you", "really", "name",
    "make", 'could', "oh", 'something', 'little', 'bit', 'life', 'feeling', 'something', 'home',
    'enough', 'sometimes', 'important',
    
}
# Join NLTK and Custom stopwords
stop_words_nltk = nltk_stop_words.union(custom_stop_words)


# Load pre-trained from 'nlp_model.ipynb'
emotion_model = load_model("models/nlp/NLP_model.h5")


# Load Whisper model for speech-to-text conversion
whisper_model = whisper.load_model("base")


# Classifies emotions of audio files using speech-to-text and the pre-trained model
def classify_audio_emotion(audio_file_path, whisper_model, emotion_model, tokenizer, stop_words, max_len=178, threshold=0.5):
    # Initialize the result dictionary
    result = {}

    # Speech-to-text using the Whisper library
    text_transcribed = whisper_model.transcribe(audio_file_path)
    text_transcribed = text_transcribed.get("text", "").strip()
    result["transcribed_text"] = text_transcribed

    # Preprocess text (clean and remove stopwords)
    text_processed = preprocess_text(text_transcribed, stop_words)

    # Text tokenize using the loaded tokenizer
    text_tokenized = tokenizer.texts_to_sequences([text_processed])

    # Validate the tokenized text
    if text_tokenized and all(token is not None for token in text_tokenized[0]):
        # Pad the sequence to match the length used during training
        text_tokenized_seq = pad_sequences(text_tokenized, maxlen=max_len, padding='post', truncating='post')

        # Predict the emotion using the pre-trained emotion model
        predictions = emotion_model.predict(text_tokenized_seq, verbose=0)

        # Define the emotion map
        emotion_map = {0: 'disgust', 1: 'fear', 2: 'anger', 3: 'joy', 4: 'sadness', 5: 'surprise'}

        # Store predicted probabilities in the result dictionary
        result["predicted_probabilities"] = {emotion: f"{prob * 100:.2f}" for emotion, prob in zip(emotion_map.values(), predictions[0])}

        # Obtain index of emotion with the highest probability
        prediction_i = np.argmax(predictions)          # Index of emotion with highest probability
        prediction_max = predictions[0][prediction_i]  # probability value

        # Determine the predicted emotion based on the threshold
        if prediction_max >= threshold:
            predicted_emotion = emotion_map[prediction_i]
        else:
            predicted_emotion = "unknown"

        # Add the predicted emotion to the result dictionary
        result["predicted_emotion"] = predicted_emotion
    else:
        # Error case for invalid tokenization
        result["error"] = "Text could not be tokenized"

    return result
