from transformers import pipeline
from PIL import Image
import numpy as np

# Load the Hugging Face pipeline for gender classification
# This will download the model the first time it is run

def load_model():
    classifier = pipeline(
        "image-classification",
        model="rizvandwiki/gender-classification"
    )
    return classifier

def preprocess_image(image):
    # No need for manual preprocessing; pipeline handles it
    return image

def predict_gender(classifier, image):
    # The pipeline expects a PIL image
    results = classifier(image)
    # Find the gender result with the highest score
    gender_results = [r for r in results if r['label'] in ['male', 'female']]
    if not gender_results:
        return "Unknown", 0.0
    best = max(gender_results, key=lambda x: x['score'])
    gender = best['label'].capitalize()
    confidence = best['score']
    return gender, confidence 