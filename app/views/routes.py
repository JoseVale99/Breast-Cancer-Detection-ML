from pyparsing import removeQuotes
from app import app
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from flask import (redirect, render_template, url_for,request, current_app,flash)
from .flask_pager import Pager
import os
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

@app.route('/')
def index():
    return render_template('index.html')

# upload file
@app.route('/upload_file', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if request.files['file'].filename == '':
            flash('¡El archivo no se cargo correctamente!','error')
            return redirect(url_for('index'))

        f = request.files['file']
        f.save(f.filename)
        df  =  pd.read_csv(f.filename)
        plot_1 = os.path.join(current_app.config['UPLOAD_FOLDER'], 'plot1.png')
        image_2 = os.path.join(current_app.config['UPLOAD_FOLDER'], 'plot2.jpg')

        flash('¡Archivo cargado exitosamente!','success')
        return render_template('index.html',plot_1= plot_1, plot_2=image_2)
    
    

# model 1
@app.route('/Regresionlogistic')
def model_logistic():
    x_test,x_train,y_train,y_test =  process_data()

    lr = LogisticRegression().fit(x_train,y_train)
    y_pred = lr.predict (x_test)
    
    id = [i for i in range (1,len(y_pred)+1)]

    data  = list(zip (id,y_pred))

    page = int(request.args.get('page', 1))

    pager = Pager(page, len(data))
    pages = pager.get_pages()

    offset = (page - 1) * current_app.config['PAGE_SIZE']
    limit = current_app.config['PAGE_SIZE']
    data_to_show = data[offset: offset + limit]  

    pred = round(accuracy_score(y_test,y_pred)*100)

    return render_template('logistic.html', pages=pages, pred=pred,data=data_to_show)

@app.route('/DecisionTreeClassifier')
def model_DecisionTreeClassifier():
    x_test,x_train,y_train,y_test =  process_data()

    # Decision Tree Classifier
    dtc = DecisionTreeClassifier()
    #Loading the training data in the model
    dtc.fit (x_train,y_train)

    #Predicting output with the test data
    y_pred = dtc.predict (x_test)
    id = [i for i in range (1,len(y_pred)+1)]

    data  = list(zip (id,y_pred))

    page = int(request.args.get('page', 1))

    pager = Pager(page, len(data))
    pages = pager.get_pages()

    offset = (page - 1) * current_app.config['PAGE_SIZE']
    limit = current_app.config['PAGE_SIZE']
    data_to_show = data[offset: offset + limit]

    pred = round(accuracy_score(y_test,y_pred)*100)

    return render_template('DecisionTreeClassifier.html',data=data_to_show,pred=pred ,pages=pages)

@app.route('/RandomForestClassifier')
def model_RandomForestClassifier():
    x_test,x_train,y_train,y_test =  process_data()

    rfc = RandomForestClassifier()
    #Loading the training data in the model
    rfc.fit (x_train,y_train)
    #Predicting output with test data
    y_pred = rfc.predict (x_test)
    id = [i for i in range (1,len(y_pred)+1)]

    data  = list(zip (id,y_pred))

    page = int(request.args.get('page', 1))

    pager = Pager(page, len(data))
    pages = pager.get_pages()

    offset = (page - 1) * current_app.config['PAGE_SIZE']
    limit = current_app.config['PAGE_SIZE']
    data_to_show = data[offset: offset + limit]

    pred = round(accuracy_score(y_test,y_pred)*100)

    return render_template('RandomForestClassifier.html',data=data_to_show,pred=pred ,pages=pages)

@app.route('/Supportvectorclassifier')
def model_support_vector_classifier():

    x_test,x_train,y_train,y_test =  process_data()

    svc = svm.SVC ()
    #Loading the training data in the model
    svc.fit (x_train,y_train)

    #Predicting output with test data
    y_pred = svc.predict (x_test)
    id = [i for i in range (1,len(y_pred)+1)]

    data  = list(zip (id,y_pred))

    page = int(request.args.get('page', 1))

    pager = Pager(page, len(data))
    pages = pager.get_pages()

    offset = (page - 1) * current_app.config['PAGE_SIZE']
    limit = current_app.config['PAGE_SIZE']
    data_to_show = data[offset: offset + limit]

    pred = round(accuracy_score(y_test,y_pred)*100)

    return render_template('Supportvectorclassifier.html',data=data_to_show,pred=pred ,pages=pages)


def process_data():
    df = pd.read_csv('data.csv')

    df=df.drop ('Unnamed: 32',axis=1)
    df ['diagnosis'].unique()
    df['diagnosis'].value_counts()
    df ['diagnosis'] = df ['diagnosis'].map ({'M':1,'B':0})
    df ['diagnosis'].unique()
    
    x = df.drop ('diagnosis',axis=1)
    y = df ['diagnosis']
    
    
    x_train,x_test,y_train,y_test = train_test_split (x,y,test_size=0.3)
    x_train.shape
    y_train.shape

    ss = StandardScaler()
    x_train = ss.fit_transform (x_train)
    x_test = ss.fit_transform (x_test)

    return (x_test,x_train,y_train,y_test)