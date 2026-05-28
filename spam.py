from datetime import datetime
from flask import Flask, render_template, request
import pickle
import os
import re
import sys

# Ensure custom text preprocessing is available while unpickling saved objects.
if __name__ != '__main__':
    sys.modules['__main__'] = sys.modules[__name__]

app = Flask(__name__, static_folder='static', template_folder='templates')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'mnb_classifier_model.pkl')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'tfidf_vectorizer.pkl')


def text_process(message):
    message = re.sub(r'[^a-zA-Z0-9\s]', ' ', message)
    tokens = re.findall(r'\b\w\w+\b', message.lower())
    return tokens


with open(VECTORIZER_PATH, 'rb') as f:
    vectorizer = pickle.load(f)

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)


LABELS = {
    0: 'Not Spam',
    1: 'Spam'
}


def classify_email(subject: str, body: str):
    content = ' '.join([subject.strip(), body.strip()]).strip()
    if not content:
        return None, None, None

    features = vectorizer.transform([content])
    prediction = int(model.predict(features)[0])
    confidence = float(model.predict_proba(features)[0][prediction]) * 100
    return LABELS.get(prediction, 'Unknown'), round(confidence, 1), prediction


@app.route('/')
def home():
    return render_template(
        'index.html',
        subject='',
        body='',
        prediction=None,
        confidence=None,
        label=None,
        timestamp=None,
        details=None
    )


@app.route('/predict', methods=['POST'])
def predict():
    subject = request.form.get('subject', '')
    body = request.form.get('body', '')
    prediction, confidence, raw_label = classify_email(subject, body)

    details = None
    if prediction is not None:
        details = (
            'This message is likely spam. Review the message carefully for unsolicited offers, suspicious links, or unexpected attachments.'
            if raw_label == 1
            else 'This message appears legitimate. Always verify sender details and avoid sharing sensitive personal information.'
        )

    return render_template(
        'index.html',
        subject=subject,
        body=body,
        prediction=prediction,
        confidence=confidence,
        label=raw_label,
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        details=details
    )


if __name__ == '__main__':
    app.run(debug=True, port=5000)
