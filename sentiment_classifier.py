import pandas as pd
import nltk
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Fixed list of stopwords (avoids NLTK issues)
stopwords_list = set([
    'a', 'an', 'the', 'and', 'or', 'in', 'on', 'with', 'to', 'of', 'for', 'by', 
    'at', 'from', 'up', 'about', 'as', 'into', 'like', 'through', 'after', 'over', 
    'between', 'out', 'against', 'during', 'without', 'before', 'under', 'around', 
    'among', 'is', 'was', 'were', 'be', 'been', 'are', 'am', 'do', 'does', 'did', 
    'have', 'has', 'had', 'this', 'that', 'these', 'those', 'it', 'he', 'she', 
    'they', 'we', 'you', 'i', 'me', 'my', 'mine', 'your', 'yours', 'his', 'her',
    'its', 'our', 'their', 'them'
])

# Load dataset
df = pd.read_csv("IMDB Dataset.csv")

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stopwords_list]
    return ' '.join(tokens)

# Progress indicator
print("Preprocessing reviews...")
df['clean_review'] = df['review'].apply(preprocess_text)

# Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_review'])
y = df['sentiment'].map({'positive': 1, 'negative': 0})

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
print("Training the model...")
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
print("Testing the model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Function to classify custom input
def classify_review(review):
    cleaned = preprocess_text(review)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)
    return "Positive" if prediction[0] == 1 else "Negative"

# Interactive loop
print("\nType a review to classify ('exit' to quit):")
while True:
    user_input = input("> ")
    if user_input.lower() in ['exit', 'quit']:
        print("Shutting down classifier.")
        break
    result = classify_review(user_input)
    print("Sentiment:", result)
