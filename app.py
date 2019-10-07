import requests
import os
import pandas as pd
from flask import Flask, render_template, request
from ocr_core import ocr_core
from flask_bootstrap import Bootstrap

app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def index():
	return render_template('index.html')

if __name__ == '__main__':
    app.run()
