import pandas as pd
import requests
import tarfile
import os
import re
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib

# --- 1. DATASET PREPARATION ---
def download_and_extract_data():
    """Downloads and extracts the IMDb dataset if not already present."""
    url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
    target_path = 'aclImdb_v1.tar.gz'
    extract_path = './'

    if os.path.exists(os.path.join(extract_path, 'aclImdb')):
        print("IMDb dataset already exists. Skipping download and extraction.")
        return

    print("Downloading IMDb dataset...")
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024

    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(target_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong during download")

    print("\nExtracting dataset...")
    with tarfile.open(target_path, 'r:gz') as tar:
        tar.extractall(path=extract_path)
    print("Extraction complete.")
    os.remove(target_path)

def load_imdb_data(data_dir='aclImdb'):
    """Loads reviews from the extracted dataset folders."""
    texts, labels = [], []
    for sentiment in ['pos', 'neg']:
        path = os.path.join(data_dir, 'train', sentiment)
        for filename in tqdm(os.listdir(path), desc=f"Loading {sentiment} reviews"):
            with open(os.path.join(path, filename), 'r', encoding='utf-8') as f:
                texts.append(f.read())
                labels.append(1 if sentiment == 'pos' else 0)
    return pd.DataFrame({'review': texts, 'sentiment': labels})

def clean_text(text):
    """A simple function to clean the review text."""
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    return text

# --- Main execution ---
download_and_extract_data()
df = load_imdb_data()
print("\nCleaning text data...")
df['review'] = df['review'].apply(clean_text)

# --- 2. MODEL TRAINING ---
print("Splitting data and training model...")
X_train, X_test, y_train, y_test = train_test_split(
    df['review'], df['sentiment'], test_size=0.2, random_state=42, stratify=df['sentiment']
)

# --- THE FINAL, DEFINITIVE FIX ---
# We are creating a custom list of stop words.
# It includes the default English stop words PLUS words specific to our dataset (domain)
# that don't carry sentiment, like 'movie', 'film', etc., and words that were
# causing spurious correlations based on our testing (e.g. 'two', 'director').
default_stop_words = TfidfVectorizer(stop_words='english').get_stop_words()
custom_stop_words = list(default_stop_words) + [
    'movie', 'film', 'review', 'story', 'character', 'characters', 'plot',
    'scene', 'scenes', 'item', 'product', 'acting', 'saw', 'watched',
    'two', 'ago', 'last', 'director', 'project', 'released', 'ok ok'
]


pipeline = Pipeline([
    # We ignore words that appear in > 70% of docs or are in our custom stop list
    ('tfidf', TfidfVectorizer(stop_words=custom_stop_words, ngram_range=(1, 2), max_df=0.7)),
    ('clf', LogisticRegression(solver='liblinear', C=10))
])

# Train the model
pipeline.fit(X_train, y_train)

# --- 3. EVALUATION & SAVING ---
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel training complete.")
print(f"Accuracy on test set: {accuracy:.4f}")

joblib.dump(pipeline, 'sentiment_pipeline.pkl')
print("Model pipeline saved as 'sentiment_pipeline.pkl'")

