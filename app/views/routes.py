from app import app
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from flask import (redirect, render_template, url_for,request, current_app,flash)
import os
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import numpy as np
import json
import plotly
import plotly.express as px

global prelog

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

# model 1 - LogisticRgression
@app.route('/Regresionlogistic')
def model_logistic():
    x_train, x_test, y_train, y_test =  process_data()

    lr = LogisticRegression()
    lr.fit(x_train,y_train)
    y_pred = lr.predict(x_test)

    ##Datos de tabla
    data_to_show, id = table_data(y_pred=y_pred, y_test=y_test)

    ###grafica
    df = chart_data(y_test=y_test, y_pred=y_pred, id=id)
    fig = px.line(df, x='id', y='valor', color='tipo', markers=True)
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    #tasa de precision
    pred = accuracy_score(y_test,y_pred)

    return render_template('logistic.html', precision=pred, data=data_to_show, graphJSON=graphJSON)

#model 2 - DecisionTreeClasifier
@app.route('/DecisionTreeClassifier')
def model_DecisionTreeClassifier():
    x_train, x_test, y_train, y_test =  process_data()

    dtc = DecisionTreeClassifier()
    dtc.fit(x_train,y_train)
    y_pred = dtc.predict(x_test)

    #datos tabla
    data_to_show, id = table_data(y_pred=y_pred, y_test=y_test)

    ###grafica
    df = chart_data(y_test=y_test, y_pred=y_pred, id=id)
    fig = px.line(df, x='id', y='valor', color='tipo', markers=True)
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    #tasa de precision
    pred = accuracy_score(y_test,y_pred)

    return render_template('DecisionTreeClassifier.html', precision=pred, data=data_to_show, graphJSON=graphJSON)

#model 3 - RandomForestClassifier
@app.route('/RandomForestClassifier')
def model_RandomForestClassifier():
    x_train, x_test, y_train, y_test =  process_data()

    rfc = RandomForestClassifier()
    rfc.fit(x_train,y_train)
    y_pred = rfc.predict(x_test)

    #datos tabla
    data_to_show, id = table_data(y_pred=y_pred, y_test=y_test)

    ###grafica
    df = chart_data(y_test=y_test, y_pred=y_pred, id=id)
    fig = px.line(df, x='id', y='valor', color='tipo', markers=True)
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    #tasa de precision
    pred = accuracy_score(y_test,y_pred)

    return render_template('RandomForestClassifier.html', precision=pred, data=data_to_show, graphJSON=graphJSON)

#model 4 - supportVectorClasifier
@app.route('/Supportvectorclassifier')
def model_support_vector_classifier():
    x_train, x_test, y_train, y_test =  process_data()

    svc = SVC()
    svc.fit(x_train,y_train)
    y_pred = svc.predict (x_test)

    #datos tabla
    data_to_show, id = table_data(y_pred=y_pred, y_test=y_test)

    ###grafica
    df = chart_data(y_test=y_test, y_pred=y_pred, id=id)
    fig = px.line(df, x='id', y='valor', color='tipo', markers=True)
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    #tasa de precision
    pred = accuracy_score(y_test,y_pred)

    return render_template('Supportvectorclassifier.html', precision=pred, data=data_to_show, graphJSON=graphJSON)

#-------->>
#datos para mostrar en la tabla
def table_data(y_pred, y_test):
    id = [i for i in range (1,len(y_pred)+1)]
    data  = list(zip (id,y_test, y_pred))
    data_to_show = data
    return data_to_show, id

#funcion para crear dataframe
def chart_data(y_test, y_pred, id):
    array = []
    for i in range(len(y_test)):
        array.append('Prediccion')

    for i in range(len(y_pred)):
        array.append('Real')

    y_testn = pd.array(y_test)
    valores = np.concatenate((y_testn, y_pred), axis=0)
    ids = np.concatenate((id,id), axis=0)
    df = pd.DataFrame({'tipo':array,
                        'id':ids,
                        'valor':valores})
    return df

def process_data():
    #Leer data.csv
    df = pd.read_csv('data.csv')
    #limpiar datos
    df=df.drop('Unnamed: 32',axis=1)
    df['diagnosis'] = df['diagnosis'].map({'M':1,'B':0})

    #dividir datos para el modelo
    x = df.drop('diagnosis', axis=1)
    y = df['diagnosis']

    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25)

    #transformar datos
    ss = StandardScaler()
    x_train = ss.fit_transform(x_train)
    x_test = ss.fit_transform(x_test)

    return(x_train, x_test, y_train, y_test)
