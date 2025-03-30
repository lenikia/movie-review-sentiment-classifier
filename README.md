# Movie Review Sentiment Classifier

## Features
- Preprocessing of text (cleaning, lowercasing, stopword removal)  
- TF-IDF vectorization  
- Naive Bayes classification (MultinomialNB)  
- Accuracy evaluation  
- Terminal-based interaction: type a review and receive sentiment prediction in real time  

## Technologies Used
- Python 3.x  
- Pandas  
- Scikit-learn  
- NLTK (only for reference stopwords, replaced with custom list)  
- Regular expressions (re)  

## Dataset
The project uses the IMDB Dataset of 50K Movie Reviews, which contains 25,000 positive and 25,000 negative reviews.  
You can find it on Kaggle: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/movie-review-sentiment-classifier.git
cd movie-review-sentiment-classifier

### 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

### 3. Install dependencies
pip install pandas scikit-learn

### 4. Place the dataset
- Download 'IMDB Dataset.csv' from Kaggle
- Place it in the project root folder

### 5. Run the project
python sentiment_classifier.py
Example Output
Preprocessing reviews...
Training the model...
Testing the model...
Accuracy: 0.85

Type a review to classify ('exit' to quit):
> I loved this movie!
Sentiment: Positive

### Accuracy
This version achieves around 85% accuracy using a simple Naive Bayes model and a fixed set of stopwords. Results may vary slightly depending on system performance and TF-IDF parameters.

### Author
Leníkia Ouana
Passionate about AI, NLP, and social impact through technology.

