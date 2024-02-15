from flask import Flask, render_template, request
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier

app = Flask(__name__)

# Load the TF-IDF vectorizer and the trained model
vectorizer = joblib.load('tfidf_vectorizer_model.joblib')
model = joblib.load('voting_classifier_model.joblib')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']

        # Apply TF-IDF vectorization to the input message
        message_tfidf = vectorizer.transform([message])

        # Make prediction
        prediction = model.predict(message_tfidf)[0]

        return render_template('result.html', message=message, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
