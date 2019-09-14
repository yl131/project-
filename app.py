from flask import Flask, render_template, request, redirect
import requests
import pandas
import simplejson as json
from bokeh.plotting import figure
from bokeh.palettes import Spectral11
from bokeh.embed import components 
import api_pull as ap
import plot

app = Flask(__name__)

app.vars={}

@app.route('/')
def main():
    return redirect('/index')

@app.route('/index',methods=['GET','POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        #request was a post
        app.vars['ticker'] = request.form['ticker']
        app.vars['results'] = ap.get_data(app.vars['ticker'])
        print("length of df: ",len(app.vars['results']))
        if len(app.vars['results']) == 0:
            return render_template('error.html')
        else:
            script, div = plot.fig(app.vars['results'], app.vars['ticker'])
            f = open('%s.txt'%(app.vars['ticker']),'w')
            f.write('Ticker: %s\n'%(app.vars['ticker']))
            f.close()
            return render_template('results.html', script=script, div=div)


if __name__ == "__main__":
    app.run(debug=True)
