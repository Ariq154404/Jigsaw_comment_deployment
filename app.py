# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from flask import Flask,render_template,url_for,request
#import pandas as pd 
import numpy as np
#import pickle
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.naive_bayes import MultinomialNB
#from sklearn.externals import joblib
#import pickle
import tensorflow
import transformers as ppb
from tensorflow import keras
#import keras
import torch
#bertmodel = pickle.load(open(r'C:\Users\Ariq\Desktop\deployment\Bert.pkl', 'rb'))
#tokenizer=pickle.load(open(r'C:\Users\Ariq\Desktop\deployment\tokenizer.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict',methods=['POST'])

def predict():
     model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
     tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
     model = model_class.from_pretrained(pretrained_weights)
     from tensorflow.keras.models import model_from_json
# load json and create model
     json_file = open(r'baseline_model.json', 'r')
     loaded_model_json = json_file.read()
     json_file.close()
     loaded_model = model_from_json(loaded_model_json)
# load weights into new model
     loaded_model.load_weights(r"baseline_model.h5")
     if request.method == 'POST':
        comment = request.form['message']
        tokenized=tokenizer.encode(comment, add_special_tokens=True,truncation=True,max_length=512)
        padded = np.array([tokenized+[0]*(415-len(tokenized))])
        attention_mask = np.where(padded != 0, 1, 0)
        ids = torch.tensor(padded)  
        mask = torch.tensor(attention_mask)
        feature=[]
        with torch.no_grad():
            last_hidden_states = model(ids, attention_mask=mask)
            feature=last_hidden_states[0][:,0,:].numpy()
        my_prediction=loaded_model.predict(feature)
    
     return render_template('result.html',prediction =my_prediction)
if __name__ == '__main__':
    app.run(debug=True)