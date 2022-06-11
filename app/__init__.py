from flask import Flask

app = Flask(__name__)

app.config['PAGE_SIZE'] = 10
app.config['VISIBLE_PAGE_COUNT'] = 4
app.config.from_object('config')


from .views import routes

