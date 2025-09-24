# machine_learning_proj_sai
IMDb Sentiment Classification (First-Year Task)
Overview
This project is part of the AIML Recruitment Task (First-Year batch).
The goal is to classify IMDb movie reviews as Positive or Negative using a simple machine learning model.
I implemented a Logistic Regression model trained on text features extracted from the IMDb dataset.

Dataset
Source: IMDb Movie Reviews dataset (IMDB Dataset.csv)

Columns:
review: The text of the review
sentiment: Label → "positive" or "negative"

Preprocessing
Converted reviews to lowercase.
Applied a negation handler:
Example: "not amazing" → "not_amazing"
Added special handling for tricky cases ("not bad" → "good", "not good" → "bad", etc.)
Used TF-IDF Vectorization (TfidfVectorizer) with:
unigrams + bigrams (ngram_range=(1,2))
English stopword removal
maximum vocabulary size = 50,000

Model
Logistic Regression
max_iter=500 to ensure convergence.

Trained using train_test_split (80% training, 20% testing).

Results
Test Accuracy: mostly from (0.903 to 0.91) based on the input

Sample Predictions:
"Absolutely wonderful acting!" → Positive ✅
"I hated the film" → Negative ✅
"The plot was not bad" → Positive ✅ (handled by negation preprocessing)

How to Run
# install dependencies
pip install pandas scikit-learn
# Run the script
python imdb_sentiment.py

Resources
Scikit-learn Logistic Regression
Text Feature Extraction
