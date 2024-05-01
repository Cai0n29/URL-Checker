import streamlit as st 
import numpy as np
import pandas as pd
import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


User = st.text_input('URL')
submit = st.button(label='Submit')

urls_data = pd.read_csv(r"C:\Users\jto07\OneDrive\Desktop\HOOK\hook\CleanedURLS.csv", encoding='utf-8')


def Tokens(f):
  Token_slash = str(f.encode('utf-8')).split('/')
  total =[]
  for i in Token_slash:
    Tokens = str(i).split('-')
    Tokens_dot = []
    for p in range(0, len(Tokens)):
      temp_Tokens = str(Tokens[p]).split('.')
      Tokens_dot = Tokens_dot + temp_Tokens
    total = total + Tokens + Tokens_dot
  total = list(set(total))
  if 'com' in total:
    total.remove('com')
  return total


y = urls_data["label"]
url_list = urls_data["domain"]


vectorizer = TfidfVectorizer(tokenizer=Tokens)
X = vectorizer.fit_transform(url_list)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=42)


logit = LogisticRegression()
logit.fit(X_train, y_train)


# PREDICTION
Predict = []
Predict.append(User)
Predict = [x.replace("http://",'') for x in Predict]
Predict = [x.replace("https://",'') for x in Predict]

if submit:
  if Predict == [""]:
     st.write("""
             <p style ="font-family:Monospace; color:black; font-size: 18px;"><b>KINDLY ENTER A LINK 🔗</p>""",unsafe_allow_html=True)
  else:
    Predict = vectorizer.transform(Predict)
    New_Predict= logit.predict(Predict)
    if New_Predict == [0]:
      st.write("THIS IS A SAFE LINK")
    elif New_Predict == [1]:
        st.write("""
             <p style ="font-family:Monospace; color:black; font-size: 18px;">🪝HOOK
             <br>
             THIS IS AN UNSAFE LINK! IT MIGHT DIRECT YOU TO A MALICOUS WEBSITE
             </p>
             """,unsafe_allow_html=True)


  
