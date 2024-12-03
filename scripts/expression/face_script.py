import pickle
import os

# Saves loaded model metadata to a pickle file.
def pickle_metadata(model_path, emotions_map, pickle_path):
    metadata = {
        "model_path": model_path,
        "emotion_labels": emotions_map
    }
    
    with open(pickle_path, 'wb') as pkl_file:
        pickle.dump(metadata, pkl_file)
    
    print(f"Model metadata saved to: {pickle_path}")

emotions_map_raf = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']
pickle_metadata("models/expression/expression_model_RAF.h5", emotions_map_raf, "models/expression/expression_model_RAF_metadata.pkl")

# Load the tokenizer pickle file (created with the 'nlp_model.ipynb' notebook)
with open('models/expression/expression_model_RAF_metadata.pkl', 'rb') as file:
    model_metadata = pickle.load(file)
