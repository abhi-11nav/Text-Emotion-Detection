# Importin tensorflow library

import tensorflow
from tensorflow import keras 
from keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import numpy as npT

# Importing text-preprocessing libraries
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# importing Flask libraries
from flask import Flask, render_template, request


def input_preprocess(input_text):
    text = re.sub('[^a-zA-Z]'," ",input_text)
    
    # Re-arranging the text format
    text = " ".join(sent.lower() for sent in text.split()) 
        
    # Applying lemmatiazation and lowering
    text = " ".join(lemmatizer.lemmatize(words) for words in text.split() if words not in stopwords.words("english"))
    
    return text

# Flask app
app = Flask(__name__, template_folder="template")

# Loading the model 
model = load_model('lstm_model.h5')

# Creating a lemmatizer object
lemmatizer = WordNetLemmatizer()

vocab_size = 14815

# Requesting input from the user 
#user_input = request.form["sentence"]


user_input = input("Enter something")
# Applying the preprocess filer
processed_input = input_preprocess(user_input)

one_hot_encoded = []

for words in processed_input.split():
    one_hot_encoded.append(one_hot(words,vocab_size)[0])
    
while(len(one_hot_encoded)!=27):
    one_hot_encoded.append(0)
    
sentiment = model.predict(np.array(one_hot_encoded).reshape(1,-1))


"""
@app.route("/",methods=["GET"])
def home_page():
    return render_template("/index.html")


@app.route("/",methods=["POST"])
def predict():
    if request.method =="POST":
        
        #Vocabulary size used to train the model 
        vocab_size = 14815
        
        # Requesting input from the user 
        user_input = request.form["sentence"]
        
        # Applying the preprocess filer
        processed_input = input_preprocess(user_input)
        
        one_hot_encoded = []
        
        for words in processed_input.split():
            one_hot_encoded.append(one_hot(words,vocab_size)[0])
            
        padded_sentences = pad_sequences(one_hot_encoded,maxlen=27,padding="post")
        
        sentiment = model.predict(padded_sentences)
        
        return render_template("/index.html", prediction_text="THIS IS THE OUTPUT"+sentiment)

if __name__ == '__main__':
    app.run(debug=True)

"""