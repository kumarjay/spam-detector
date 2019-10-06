from flask import Flask, render_template, url_for, request
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib


filename= 'nlp_model.pkl'
clf= pickle.load(open(filename, 'rb'))
cv= pickle.load(open('transform.pkl', 'rb'))
app= Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
#    df= pd.read_csv('G:\dAtAsS\spam_detection\spam.csv', encoding='latin-1')
#    
#    df['label']= df['Label'].map({'ham':0, 'spam':1})
#    
#    x=df['EmailText']
#    y=df['label']
#    
#    cv= CountVectorizer()
#    X= cv.fit_transform(x)
#    
#    pickle.dump(cv, open('transform.pkl','wb'))
#    
#    
#    from sklearn.model_selection import train_test_split
#    
#    x_train, x_test, y_train, y_test= train_test_split(X, y, test_size=.2, random_state=42)
#    
#    clf= MultinomialNB()
#    clf.fit(x_train, y_train)
#    clf.score(x_test, y_test)
#    
#    
#    filename= 'nlp_model.pkl'
#    pickle.dump(clf, open(filename, 'wb'))
    
    if request.method=='POST':
        message= request.form['message']
        data=[message]
        vect= cv.transform(data).toarray()
        my_prediction= clf.predict(vect)
        
    return render_template('result.html', prediction= my_prediction)


if __name__=='__main__':
    app.run(debug=True)