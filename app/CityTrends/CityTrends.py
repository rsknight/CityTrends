#### ------- app_CT ------- ####
from flask import render_template
from flask import request
from flaskexample import app
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import numpy as np
import psycopg2
import json
import plotly
import plotly.graph_objs as go
import plotly.express as px
import pickle
from fbprophet import Prophet 

user = 'postgres' #add your username here (same as previous postgreSQL)   
password = '5802'                   
host = 'localhost'
dbname = 'phillyRE_db'
db = create_engine('postgres://%s%s/%s'%(user,host,dbname))
con = None
con = psycopg2.connect(database = dbname, user = user, host = host, password = password)

start_date = '1999-01-01'
end_date = '2019-12-31'

@app.route('/')
@app.route('/index')
def philly_realestate_input():
    return render_template("index.html")


@app.route('/output')
def philly_realestate_output():
  # with open("/mnt/Ravenclaw/Insight/app_CT/flaskexample/saved_forecast", "r") as f: my_loaded_forecaster = pickle.load(f)	
  # read the Prophet model object
 

  #pull 'zip code' from input field and store it
  zips = request.args.get('zip_code')
  query = "SELECT * FROM philly_realestate_table WHERE zip_code='%s' AND document_type='DEED' AND cash_consideration > 1"  % zips
  print(query)
  query_results=pd.read_sql_query(query,con)
  print(query_results)
  # Convert to a date first
  query_results['display_date'] = query_results['display_date'].astype('datetime64[ns]')
  #Then sort...
  df_sorted = query_results.sort_values(by='display_date')
  # Only include from year 1999
  mask = (df_sorted['display_date'] > start_date) & (df_sorted['display_date'] <= end_date)
  df_zip_temp = df_sorted.loc[mask] # Temp (before outlier removal)
  # Remove outliers 
  q_zip_upper = df_zip_temp["cash_consideration"].quantile(0.95)
  q_zip_lower = df_zip_temp["cash_consideration"].quantile(0.05)
  q_mask = (df_zip_temp['cash_consideration'] <  q_zip_upper) & (df_zip_temp['cash_consideration'] >  q_zip_lower)
  df_zip = df_zip_temp.loc[q_mask]
  
  # New dataframe labels for
  data = {'ds': df_zip.display_date, 'y': df_zip.cash_consideration}
  df = pd.DataFrame(data)
  
  #~ #Create graph
  df['dates'] = pd.to_datetime(df.ds)
  new_means = df.groupby(df.dates.dt.year)['y'].transform('mean')
  # data = [px.scatter(df, x=df['display_date'], y=df['cash_consideration'])]
  
  data = [go.Scatter(x=df_zip['display_date'], y=new_means)]
  graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
  return render_template('output.html', graphJSON=graphJSON)  

