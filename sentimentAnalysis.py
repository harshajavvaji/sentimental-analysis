from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from typing import Union

import re
from nltk.tokenize.treebank import TreebankWordDetokenizer
import gensim
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import keras
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

app = FastAPI()

@app.get("/")
def home_page():
    return {"Message": "Sentiment Analysis API"}

# Sentiment Analysis
train = pd.read_csv(r'sentimentAnalysisData\train.csv')

#Is there any other different value than neutral, negative and positive?
train['sentiment'].unique()

#How's distributed the dataset? Is it biased?
train.groupby('sentiment').nunique()

#Let's keep only the columns that we're going to use
train = train[['selected_text','sentiment']]

#Is there any null value?
train["selected_text"].isnull().sum()

#Let's fill the only null value.
train["selected_text"].fillna("No content", inplace = True)

def depure_data(data):
    #Removing URLs with a regular expression
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    data = url_pattern.sub(r'', data)

    # Remove Emails
    data = re.sub('\S*@\S*\s?', '', data)

    # Remove new line characters
    data = re.sub('\s+', ' ', data)

    # Remove distracting single quotes
    data = re.sub("\'", "", data)
    return data

temp = []
data_to_list = train['selected_text'].values.tolist()
for i in range(len(data_to_list)):
    temp.append(depure_data(data_to_list[i]))
list(temp[:5])

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data_words = list(sent_to_words(temp))

def detokenize(text):
    return TreebankWordDetokenizer().detokenize(text)

data = []
for i in range(len(data_words)):
    data.append(detokenize(data_words[i]))

data = np.array(data)

labels = np.array(train['sentiment'])
y = []
for i in range(len(labels)):
    if labels[i] == 'neutral':
        y.append(0)
    if labels[i] == 'negative':
        y.append(1)
    if labels[i] == 'positive':
        y.append(2)
y = np.array(y)
labels = tf.keras.utils.to_categorical(y, 3, dtype="float32")
del y

max_words = 5000
max_len = 200

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(data)
sequences = tokenizer.texts_to_sequences(data)
tweets = pad_sequences(sequences, maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(tweets,labels, random_state=0)

best_model = keras.models.load_model(r"sentimentAnalysisData\biderectionModel.hdf5")
print("Model Loaded")

class TextContent(BaseModel):
    text: Union[str, None] = None

@app.post("/sentiment_analysis/")
async def sentiment_analysis(text_ref: TextContent):
    sentiment = ['Neutral','Negative','Positive']
    try:
        sequence = tokenizer.texts_to_sequences([text_ref.text])
        test = pad_sequences(sequence, maxlen=max_len)
        textSentiment = sentiment[np.around(best_model.predict(test), decimals=0).argmax(axis=1)[0]]
    except Exception as e:
        data = {
            "status": e,
        }
    data = {
        'textSentiment': textSentiment
    }
    return data

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)