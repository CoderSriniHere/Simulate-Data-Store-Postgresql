import json
import random
import numpy as np
import nltk
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import gradio as gr

# Initialize
stemmer = PorterStemmer()
nltk.download('punkt')

# Load intents
with open("intents.json") as file:
    data = json.load(file)

# Preprocessing
all_words = []
tags = []
xy = []

for intent in data['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        tokens = nltk.word_tokenize(pattern)
        stemmed = [stemmer.stem(w.lower()) for w in tokens]
        all_words.extend(stemmed)
        xy.append((" ".join(stemmed), tag))

# Vectorization
corpus = [pattern for (pattern, tag) in xy]
labels = [tag for (pattern, tag) in xy]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
y = np.array(labels)

# Train model
model = MultinomialNB()
model.fit(X, y)

# Define the chatbot response function for Gradio
def get_response(user_input):
    if not user_input.strip():
        return "Please say something :)"

    tokens = nltk.word_tokenize(user_input)
    stemmed = [stemmer.stem(w.lower()) for w in tokens]
    input_text = " ".join(stemmed)
    input_vec = vectorizer.transform([input_text])

    predicted_tag = model.predict(input_vec)[0]

    for intent in data['intents']:
        if intent['tag'] == predicted_tag:
            response = random.choice(intent['responses'])
            return response

    return "Sorry, I didn't understand that. Can you ask differently?"

# Create Gradio interface
iface = gr.Interface(
    fn=get_response,
    inputs=gr.Textbox(lines=2, placeholder="Type your message here..."),
    outputs="text",
    title="Dance Chatbot",
    description="Ask me anything about dance or just chat!"
)

# Launch with share=True to get a public URL
iface.launch(share=True)
