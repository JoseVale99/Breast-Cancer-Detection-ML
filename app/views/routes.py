from app import app
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from flask import (redirect, render_template, request, current_app,flash, Blueprint, url_for)
from .flask_pager import Pager
import os
import io
import base64


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
        image = os.path.join(current_app.config['UPLOAD_FOLDER'], 'plot1.png')

        flash('¡Archivo cargado exitosamente!','success')
        return render_template('index.html',image=image)
    
    

# model 1
@app.route('/Regresionlogistic')
def model_logistic():
    df=pd.read_csv ("data/data.csv")

    df=df.drop ('Unnamed: 32',axis=1)
    #df ['diagnosis'].unique()
    #df['diagnosis'].value_counts()
    df ['diagnosis'] = df['diagnosis'].map ({'M':1,'B':0})
    #df ['diagnosis'].unique()

    x = df.drop('diagnosis', axis=1)
    y = df['diagnosis']
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25)

    from sklearn.preprocessing import StandardScaler
    ss = StandardScaler()
    x_train = ss.fit_transform(x_train)
    x_test = ss.fit_transform(x_test)

    from sklearn.linear_model import LogisticRegression
    logr = LogisticRegression()
    logr.fit(x_train, y_train)
    y_pred = logr.predict(x_test)

    ##Datos de tabla
    id = [i for i in range (1,len(y_pred)+1)]
    data  = list(zip (id,y_test, y_pred))
    page = int(request.args.get('page', 1))
    pager = Pager(page, len(data))
    pages = pager.get_pages()
    offset = (page - 1) * current_app.config['PAGE_SIZE']
    limit = current_app.config['PAGE_SIZE']
    data_to_show = data[offset: offset + limit]
    ###

    from sklearn.metrics import accuracy_score
    precision = accuracy_score(y_test, y_pred)

    image = os.path.join(current_app.config['UPLOAD_FOLDER'], 'plotlog.png')

    '''plot = sns.countplot(y_pred)

    x_1 = [1,3,5,7,9,10]
    y_1 = [6,9,12,15,18,21]
    x_2 = [-6,-4,-2,0,2,3]
    y_2 = [-1,2,5,8,11,14]

    ids = [1]
    for i in range(2, 172):
        ids.append(i)

    y_test = pd.array(y_test)

    plt.clf()
    img = io.BytesIO()
    plt.plot(ids,y_test,marker='.',color='darkred',linestyle='-')
    plt.plot(ids,y_pred,marker='.',color='darkblue',linestyle='-')
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()   
    return render_template('logistic.html', imagen={ 'imagen': plot_url })'''

    


    '''X = df.iloc[:, 2:31].values
    Y = df.iloc[:, 1].values
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)

    from sklearn.linear_model import LogisticRegression
    log = LogisticRegression(random_state=0)
    log.fit(X_train, Y_train)
    precision = log.score(X_train, Y_train)

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(Y_test, log.predict(X_test))
    TN = cm[0][0]
    TP = cm[1][1]
    FN = cm[1][0]
    FP = cm[0][1]

    precisionm = (TP + TN)/(TP + TN + FN + FP)'''
    
    '''x = df.drop ('diagnosis',axis=1)
    y = df ['diagnosis']
    
    
    x_train,x_test,y_train,y_test = train_test_split (x,y,test_size=0.3)
    x_train.shape
    y_train.shape
        
    ss = StandardScaler()
    x_train = ss.fit_transform (x_train)
    x_test = ss.fit_transform (x_test)

    lr = LogisticRegression().fit(x_train,y_train)
    y_pred = lr.predict (x_test)
    
    id = [i for i in range (1,len(y_pred)+1)]

    data  = list(zip (id,y_pred))

    page = int(request.args.get('page', 1))

    pager = Pager(page, len(data))
    pages = pager.get_pages()

    offset = (page - 1) * current_app.config['PAGE_SIZE']
    limit = current_app.config['PAGE_SIZE']
    data_to_show = data[offset: offset + limit]'''    

    #return render_template('logistic.html', pages=pages, data=data_to_show)
    return render_template('logistic.html', precision=precision, imagen=image, pages=pages, data=data_to_show)