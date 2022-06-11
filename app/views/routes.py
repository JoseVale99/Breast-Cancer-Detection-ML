from app import app
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from flask import render_template, request, current_app
from .flask_pager import Pager


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/Regresionlogistic')
def model_logistic():
    df=pd.read_csv ("data/data.csv")
    df.isna ().sum()
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

    lr = LogisticRegression().fit(x_train,y_train)
    y_pred = lr.predict (x_test)
    
    id = [i for i in range (1,len(y_pred)+1)]

    data  = list(zip (id,y_pred))

    page = int(request.args.get('page', 1))

    # count = 300
    # data_1 = range(count)

    pager = Pager(page, len(data))
    pages = pager.get_pages()

    offset = (page - 1) * current_app.config['PAGE_SIZE']
    limit = current_app.config['PAGE_SIZE']
    data_to_show = data[offset: offset + limit]    

    return render_template('logistic.html', pages=pages, data=data_to_show)