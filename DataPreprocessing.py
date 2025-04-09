import re
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class TextPreprocessor:
    def __init__(self, use_tfidf=False):
        """
        Initializes the TextPreprocessor.
        :param use_tfidf: If True, enables TF-IDF vectorization.
        """
        self.use_tfidf = use_tfidf
        self.vectorizer = TfidfVectorizer() if use_tfidf else None
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text):
        """
        Cleans the text by converting to lowercase, removing special characters and extra spaces.
        """
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)  # Keep only letters and spaces
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
        return text

    def tokenize(self, text):
        """
        Tokenizes the text into words.
        """
        return word_tokenize(text)

    def remove_stopwords(self, tokens):
        """
        Removes stopwords from the tokenized text.
        """
        return [word for word in tokens if word not in self.stop_words]

    def lemmatize(self, tokens):
        """
        Lemmatizes tokens to their base form.
        """
        return [self.lemmatizer.lemmatize(word) for word in tokens]

    def preprocess(self, text):
        """
        Full preprocessing pipeline: clean, tokenize, remove stopwords, and lemmatize.
        """
        text = self.clean_text(text)
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        tokens = self.lemmatize(tokens)
        return " ".join(tokens)

    def fit_tfidf(self, texts):
        """
        Fits the TF-IDF vectorizer on the given list of texts and transforms them.
        """
        if self.use_tfidf:
            return self.vectorizer.fit_transform(texts).toarray()
        else:
            raise ValueError("TF-IDF is not enabled. Initialize with use_tfidf=True.")

# Example Usage
if __name__ == "__main__":
    sample_texts = [
        "The overall goal of this research was to determine ways to do better.",
        "Motivated students learn better when they understand the importance of the material."
    ]
    
    preprocessor = TextPreprocessor(use_tfidf=True)
    
    # Preprocess text
    cleaned_texts = [preprocessor.preprocess(text) for text in sample_texts]
    print("Cleaned Texts:", cleaned_texts)
    
    # Convert to TF-IDF (if enabled)
    tfidf_vectors = preprocessor.fit_tfidf(cleaned_texts)
    print("TF-IDF Vectors:", tfidf_vectors)
