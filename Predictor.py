import pickle
import numpy as np
from DataPreprocessing import TextPreprocessor
from Model import models, vectorizer  # Assuming models and vectorizer are saved in Models.py

def predict_text(text):
    # Initialize Text Preprocessor
    preprocessor = TextPreprocessor(use_tfidf=True)
    processed_text = preprocessor.preprocess(text)
    
    # Transform text using the trained TF-IDF vectorizer
    text_vector = vectorizer.transform([processed_text])
    
    predictions = {}
    for name, model in models.items():
        prediction = model.predict(text_vector)[0]
        predictions[name] = "Human" if prediction == 1 else "AI"
    
    return predictions

if __name__ == "__main__":
    user_text = input("Enter text to predict whether it's AI or human-generated: ")
    results = predict_text(user_text)
    
    for model, prediction in results.items():
        print(f"{model}: {prediction}")
