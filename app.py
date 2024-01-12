from flask import Flask,render_template
from flask import request
import pickle,numpy as np,pandas as pd,nltk,string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
from sklearn.feature_extraction.text import TfidfVectorizer
model=pickle.load(open('model.pkl','rb'))
tfidf=pickle.load(open('tfidf.pkl','rb'))
def transformed_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)  
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)  
    text=y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)
app=Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict_toxic():
    text=request.form.get('text')
    text=transformed_text(text)
    df=pd.DataFrame([text])
    df=tfidf.transform(df[:][0])
    result=model.predict(df)
    return render_template('index.html',result=result[0])
if __name__=='__main__':
    app.run(debug=True)