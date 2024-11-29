from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle 
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt_tab')
nltk.download('stopwords')

app = Flask(__name__)
CORS(app) 

with open('decision_tree_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

stop_words = set(stopwords.words('english'))
def preprocess_tweet(text):
    if isinstance(text, str):
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'[^A-Za-z\s]', '', text.lower())
        tokens = word_tokenize(text)
        filtered_tokens = [word for word in tokens if word not in stop_words]
        return ' '.join(filtered_tokens)
    else:
        return ''



@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    tweet = data.get('tweet', '')
    if not tweet:
        return jsonify({'error': 'No tweet provided'}), 400

    processed_tweet = preprocess_tweet(tweet)
    X_train_tfidf = vectorizer.transform([processed_tweet])
    prediction = model.predict(X_train_tfidf)
    sentiment = prediction[0]
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)
