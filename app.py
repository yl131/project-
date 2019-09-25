import requests
import pandas as pd
from flask import Flask, render_template, request
from bokeh.plotting import figure
from bokeh.embed import components

app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def index():
	return render_template('index.html')



if __name__ == '__main__':
	app.run(port=33507)
