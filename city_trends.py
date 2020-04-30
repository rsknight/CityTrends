import pandas as pd 
import psycopg2
import numpy as np
import matplotlib.pyplot as plt
from fbprophet import Prophet 
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
from geoalchemy2 import Geometry, WKTElement
from pandas.plotting import register_matplotlib_converters
from pandas.tseries import converter
import geopandas as gpd
import datetime as dt

# Define database
user = 'postgres'
password =   
host = 'localhost'
port = '5432'  
dbname = 'phillyRE_db'
engine = create_engine( 'postgresql://{}:{}@{}:{}/{}'.format(user, password, host, port, dbname) )
print(engine.url)

# Read CSV into pandas dataframe
phillyRE_data = pd.read_csv('RTT_SUMMARY_downsampled.csv', index_col=False)

# Insert data into SQL database
phillyRE_data.to_sql('philly_realestate_table', engine, if_exists='replace')

# Connect to make queries using psycopg2
con = None
con = psycopg2.connect(database = dbname, user = user, host = host, password = password)

# Query 
sql_query = """
SELECT * FROM philly_realestate_table 
WHERE document_type='DEED'
AND cash_consideration > 30000
AND property_count = 1
AND street_address is not null
AND lat is not null 
AND lng is not null;"""
data = pd.read_sql_query(sql_query,con)

# Inspect data 
data.info()
# Check if there's any missing data
data.isnull().sum()
# Column descriptive stats 
data.describe()

# Remove price upper outliers 
data_temp_upper = data
q_upper = data_temp_upper["cash_consideration"].quantile(0.99)
q_mask = (data_temp_upper['cash_consideration'] <  q_upper)
data_price_temp = data_temp_upper.loc[q_mask]
data_price_temp.describe()
data = data_price_temp

# Sort by date 
data['display_date'] = data['display_date'].astype('datetime64[ns]')
date_sorted = data.sort_values(by='display_date')

# Only include from year 1999
start_date = '1999-01-01'
end_date = '2019-12-31'
mask_date = (date_sorted['display_date'] >= start_date) & (date_sorted['display_date'] <= end_date)
data = date_sorted.loc[mask_date]


# VISUALIZE DATA
# Sales Price 
plt.hist(data['cash_consideration'], edgecolor='black', color='mediumvioletred')
plt.show()

# Data are super skewed, let's see if log function gives a normal distribution 
log_price = np.log(data['cash_consideration'])
plt.hist(log_price, edgecolor='black', color='mediumturquoise')
plt.show()
# It does, so let's use it

# 1-- Read income data into dataframe 
philly_income_data = pd.read_csv('ACS_5YR_median_income.csv', index_col=False)
# Remove un-needed words in tract column (keep only tract number)
philly_income_data['GEO.display-label'] = philly_income_data['GEO.display-label'].map(lambda x: x.lstrip('Census Tract ').rstrip(', Philadelphia County, Pennsylvania'))
# Change column names to median_income and census_tract
philly_income_data=philly_income_data.rename(index=str, columns={"GEO.display-label":"census_tract", "Estimate":"median_income", "Year":"census_year"})
philly_income_data = philly_income_data.drop(['GEO.id','GEO.id2'], axis = 1)
philly_income_data = philly_income_data[~philly_income_data['median_income'].isin(['-'])]
philly_income_data['median_income'] = philly_income_data['median_income'].astype(np.int64)
# Convert census tract from object to float 
philly_income_data['census_tract'] = philly_income_data['census_tract'].astype(np.float)

# Remove outliers 
df_philly_income_temp = philly_income_data
q_lower_income = df_philly_income_temp['median_income'].quantile(0.10)
q_mask_income = (df_philly_income_temp['median_income'] >  q_lower_income)
philly_income_data = df_philly_income_temp.loc[q_mask_income]

# 2-- Get census tract geoData
# Load JSON file
gdf_tracts = gpd.read_file('Philadelphia_Census_Tracts_2010_201302.geojson') 
# Rename geoDtaframe columns to match pandas(home)Dataframe columns (not sure if necessary) 
gdf_tracts=gdf_tracts.rename(index=str, columns={"INTPTLAT10":"lat", "INTPTLON10":"lng", "NAMELSAD10":"census_tract"})
gdf_tracts['census_tract'] = gdf_tracts['census_tract'].map(lambda x: x.lstrip('Census Tract '))

# 3-- Convert real estate pandas dataframe into geopandas dataframe 
gdf_sales = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.lng, data.lat))
# Make CRS same as geodataframe 
gdf_sales.crs = {'init' :'epsg:4326'}
# Join the real estate and census tract dataframes 
gdf_sales_tracts = gpd.sjoin(gdf_sales, gdf_tracts, op='within')
# Create new column (census_year) based on sales date (display_date)
gdf_sales_tracts['census_year'] = gdf_sales_tracts['display_date'].dt.year
# Convert census tract to float (to match with income and education)
gdf_sales_tracts['census_tract'] = gdf_sales_tracts['census_tract'].astype(np.float)

#4 -- Get Education data 
census_edu = pd.read_csv('census_edu.csv')
del census_edu['Unnamed: 0']

# Join the sales and income dataframes 
gdf_sales_income = gdf_sales_tracts.merge(philly_income_data, on=['census_tract','census_year'])
# Join the education data
gdf_sales_income_edu = gdf_sales_income.merge(census_edu, on=['census_tract','census_year'])
# Drop extra education columns 
gdf_sales_income_edu = gdf_sales_income_edu.drop(['Total_Pop_over25', 'Pop_over25_Bachelor'], axis = 1)
# # Turn census tract column into object 
gdf_sales_income_edu['census_tract'] = gdf_sales_income_edu['census_tract'].astype(str)
gdf_sales_income_edu.shape

# Get census tract map 
plt.rcParams['figure.figsize'] = (20, 10)   # Resize map figure
ax = gdf_tracts.plot(color='white', edgecolor='black')

# Plot real estate sales locations  
plt.rcParams['figure.figsize'] = (20, 10)   # Resize map figure, default is really small 
new_ax = gdf_sales.plot(markersize=1, color='red')

# Plot census tract map and real estate data
plt.rcParams['figure.figsize'] = (20, 10)
base_map = gdf_tracts.plot(color='white', edgecolor='black')
gdf_sales.plot(ax=base_map, markersize=1, color='red', alpha=0.05) #with transparency


# Fitting model with fbProphet -- first get income forecast
df_income_ds_y = pd.DataFrame({'ds': gdf_sales_income_edu.display_date, 'y': gdf_sales_income_edu.median_income})
model_income = Prophet()
model_income.fit(df_income_ds_y)
future_income = model_income.make_future_dataframe(periods=1825)
forecast_income = model_income.predict(future_income)
model_income.plot(forecast_income,xlabel='Year', ylabel='Median Income');

# Next, get education forecast
df_edu_ds_y = pd.DataFrame({'ds': gdf_sales_income_edu.display_date, 'y': gdf_sales_income_edu['perc_Bachelor']})
model_edu = Prophet()
model_edu.fit(df_edu_ds_y)
future_edu = model_edu.make_future_dataframe(periods=1825)
forecast_edu = model_edu.predict(future_edu)
model_edu.plot(forecast_edu,xlabel='Year', ylabel='Educational Attaninment');

# Then, add them to main sales data as regressors
df_city_trends = pd.DataFrame(columns=['ds','y','income'])
df_city_trends['ds'] = gdf_sales_income_edu.display_date
df_city_trends['y'] = gdf_sales_income_edu.cash_consideration
df_city_trends['income'] = forecast_income.trend
df_city_trends['edu'] = forecast_edu.trend

# Model 
model_city_trends = Prophet()
model_city_trends.add_regressor('income')
model_city_trends.add_regressor('edu')
model_city_trends.fit(df_city_trends)

# Forecasting 
future_city_trends = model_city_trends.make_future_dataframe(periods=1825, freq='D')
future_city_trends['income'] = df_city_trends.income
future_city_trends['income'].iloc[-1825:] = forecast_income.trend
future_city_trends['edu'] = df_city_trends.edu
future_city_trends['edu'].iloc[-1825:] = forecast_edu.trend
forecast_city_trends = model_city_trends.predict(future_city_trends)
model_city_trends.plot(forecast_city_trends,xlabel='Year', ylabel='Sales Price');

# Evaluate trends 
figure_trends = model_city_trends.plot_components(forecast_city_trends)

# Validation
from fbprophet.diagnostics import cross_validation
gdf_cv = cross_validation(model_city_trends, initial='2555 days', period='180 days', horizon = '730 days')
# Diagnostics 
from fbprophet.diagnostics import performance_metrics
gdf_p = performance_metrics(gdf_cv)
from fbprophet.plot import plot_cross_validation_metric
fig = plot_cross_validation_metric(gdf_cv, metric='mape')

# Save model 
import pickle
with open('forecast_city_trends_up.pckl', 'wb') as fout:
    pickle.dump(model_city_trends, fout)
with open('forecast_city_trends_up.pckl', 'rb') as fin:
    saved_model = pickle.load(fin)
