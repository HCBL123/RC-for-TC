import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from reservoir import ReservoirNetwork
import re
from collections import Counter
from tqdm import tqdm
from datasets import load_dataset

# Updated emotion words dictionary for the dair-ai dataset classes
EMOTION_WORDS = {
    'sadness': {
        'high': ['devastated', 'heartbroken', 'miserable', 'depressed', 'grief'],
        'medium': ['sad', 'unhappy', 'hurt', 'disappointed', 'sorry'],
        'low': ['down', 'blue', 'miss', 'meh', 'sigh']
    },
    'joy': {
        'high': ['ecstatic', 'overjoyed', 'thrilled', 'fantastic', 'amazing'],
        'medium': ['happy', 'joy', 'glad', 'pleased', 'wonderful'],
        'low': ['content', 'satisfied', 'good', 'nice', 'okay']
    },
    'love': {
        'high': ['adore', 'passionate', 'cherish', 'soulmate', 'beloved'],
        'medium': ['love', 'loving', 'affection', 'care', 'sweet'],
        'low': ['like', 'fond', 'warm', 'appreciate', 'crush']
    },
    'anger': {
        'high': ['furious', 'enraged', 'hate', 'outraged', 'livid'],
        'medium': ['angry', 'mad', 'frustrated', 'annoyed', 'irritated'],
        'low': ['bothered', 'dislike', 'upset', 'grumpy', 'miffed']
    },
    'fear': {
        'high': ['terrified', 'panicked', 'horrified', 'petrified', 'dread'],
        'medium': ['scared', 'afraid', 'fearful', 'anxious', 'worried'],
        'low': ['nervous', 'uneasy', 'concerned', 'apprehensive', 'cautious']
    },
    'surprise': {
        'high': ['shocked', 'astounded', 'astonished', 'amazed', 'stunned'],
        'medium': ['surprised', 'unexpected', 'startled', 'wow', 'whoa'],
        'low': ['unusual', 'strange', 'odd', 'different', 'curious']
    }
}

def preprocess_text(text):
    """Enhanced text preprocessing"""
    # Convert to lowercase, keeping original for capitalization feature
    caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
    text = str(text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove user mentions and hashtags but keep hashtag content
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Replace repeated characters (e.g., 'happyyyy' -> 'happy')
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    # Store punctuation counts
    exclamations = text.count('!')
    questions = text.count('?')
    
    # Remove numbers and special characters, keeping only letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    text = text.strip().lower()
    
    # Add back structural markers
    text = text + ' ' + '!' * min(exclamations, 3)
    text = text + ' ' + '?' * min(questions, 3)
    if caps_ratio > 0.5:
        text = text + ' ' + 'emphasis'
    
    return text

def get_emotion_intensity(text, emotion):
    """Calculate emotion intensity based on word presence and patterns"""
    intensity = 0
    words = set(text.lower().split())
    
    if emotion in EMOTION_WORDS:
        for level, word_list in EMOTION_WORDS[emotion].items():
            weight = 3 if level == 'high' else 2 if level == 'medium' else 1
            intensity += sum(word in words for word in word_list) * weight
    
    # Additional intensity from patterns
    if emotion == 'joy' or emotion == 'love':
        intensity += text.count('!') * 0.5
        intensity += text.count('<3') * 2
    elif emotion == 'anger':
        intensity += text.count('!') * 0.5
        intensity += text.upper() == text and len(text) > 3
    elif emotion == 'surprise':
        intensity += text.count('!') * 0.5
        intensity += text.count('?') * 0.5
    
    return intensity

def count_emotion_words(text):
    """Enhanced emotion word counting with intensity features"""
    features = {}
    
    # Calculate intensity for each emotion
    for emotion in EMOTION_WORDS.keys():
        features[f"{emotion}_intensity"] = get_emotion_intensity(text, emotion)
    
    # Additional structural features
    features['exclamation_marks'] = text.count('!')
    features['question_marks'] = text.count('?')
    features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
    
    return features

def extract_features(texts, vectorizer=None, is_training=True):
    """Extract features and convert to GPU tensor"""
    # TF-IDF features
    if is_training:
        tfidf_features = vectorizer.fit_transform(texts).toarray()
    else:
        tfidf_features = vectorizer.transform(texts).toarray()
    
    # Emotion word count features
    emotion_features = []
    for text in tqdm(texts, desc="Extracting emotion features"):
        features = count_emotion_words(text)
        emotion_features.append(list(features.values()))
    
    # Combine features
    X = np.hstack([tfidf_features, np.array(emotion_features)])
    
    # Convert to float32 for GPU efficiency
    X = X.astype(np.float32)
    
    return X

def load_data():
    """Load and preprocess the emotion dataset"""
    print("Loading data from dair-ai/emotion dataset...")
    ds = load_dataset("dair-ai/emotion", "unsplit")
    
    # Convert to pandas DataFrame for easier processing
    df = pd.DataFrame(ds['train'])
    
    print("Preprocessing text data...")
    df['text'] = df['text'].apply(preprocess_text)
    
    # Remove empty texts after preprocessing
    df = df[df['text'].str.strip().astype(bool)]
    
    return df['text'].values, df['label'].values

def print_class_distribution(y):
    """Print class distribution in the dataset"""
    unique, counts = np.unique(y, return_counts=True)
    distribution = dict(zip(unique, counts))
    print("\nClass distribution:")
    for label in sorted(distribution.keys()):
        count = distribution[label]
        percentage = (count / len(y)) * 100
        print(f"{label}: {count} ({percentage:.1f}%)")

def main():
    # Enable CUDA optimizations
    torch.backends.cudnn.benchmark = True
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X, y = load_data()
    
    # Print initial class distribution
    print("\nInitial class distribution:")
    print_class_distribution(y)
    
    # Initialize TF-IDF vectorizer
    print("\nInitializing feature extraction...")
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9
    )
    
    # Extract features
    print("Extracting features...")
    X_features = extract_features(X, vectorizer, is_training=True)
    
    # Split data
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Initialize and train reservoir network
    print("Training reservoir network...")
    reservoir = ReservoirNetwork(
        input_dim=X_train.shape[1],
        reservoir_size=400,
        spectral_radius=0.95,
        sparsity=0.1,
        noise=0.001
    )
    
    # Train with GPU acceleration
    reservoir.train(X_train, y_train, batch_size=64, epochs=10)
    
    # Make predictions
    print("Making predictions...")
    y_pred = reservoir.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(
        y_test, 
        y_pred,
        zero_division=0
    ))
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Save the trained model
    torch.save({
        'model_state_dict': reservoir.readout.state_dict(),
        'vectorizer': vectorizer,
    }, 'models/emotion_classifier.pt')

if __name__ == "__main__":
    main()