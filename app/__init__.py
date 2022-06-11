from flask import Flask
import os
app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
app.config['PAGE_SIZE'] = 10
app.config['VISIBLE_PAGE_COUNT'] = 4
app.config.from_object('config')
app.config['UPLOAD_FOLDER'] =  os.path.join('static')

from .views import routes

