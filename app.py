# import libraries
import streamlit as slt
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import datetime
from datetime import date,timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from  statsmodels.tsa.stattools import adfuller
import streamlit as st




# Title of Application
app_name = "Stock Market Forecasting App"
slt.title(app_name)
slt.subheader("This app is created to Forecast the Stock market of the selected companies")

#add an image from online resourse
slt.image("https://img.freepik.com/free-vector/gradient-stock-market-concept_23-2149166910.jpg")

#Taake input from the user of app about start and end date

#sidebar
slt.sidebar.header("Select the Parameter from Below")
start_date=slt.sidebar.date_input("Start Date",date(2020,1,1))
end_date=slt.sidebar.date_input("Start Date",date(2023,8,31))
#add ticker symbol list
ticker_list=["AAPL","MSFT","GOOG","GOOGL","TSLA","META","NVDA","ADBE","PYPL","INTC","CMCSA","NFLX","PEP"]
ticker=slt.sidebar.selectbox("Select Company Name",ticker_list)

#fetch data from user input using  yfinance Library
data=yf.download(ticker,start=start_date,end=end_date)
#add date as column in dataframe
data.insert(0,"Date",data.index,True)
data.reset_index(drop=True,inplace=True)
slt.write('Data from',start_date,'to',end_date)
slt.write(data)

#plot the data
slt.header("Data Visualization")
slt.subheader("Plot the Data")
slt.write("**Note** Select your date range from sidebar or zoom in on the plot and select your desired column")
fig=px.line(data,x='Date',y=data.columns,title='closing price of stock',width=800,height=600)
slt.plotly_chart(fig)

#add a select box to select data from column
column=slt.selectbox("Select the column to be used for forecasting",data.columns[1:])

#subsetting the data
data=data[['Date',column]]
slt.write("Seleted Data")
slt.write(data)




# ADF test Check stationarity
st.header("Is Data Stationary")
#st.write("**Note** if p value is less then 0.05,the data is stationary")
# Assuming you have a DataFrame 'data' and a column 'column' defined elsewhere
st.write(adfuller(data[column])[1] < 0.05)

#Lets decompose the data
st.header("Decompose the Data")
decomposition=seasonal_decompose(data[column],model='additive',period=12)
st.write(decomposition.plot())

#make same plot in ploty
st.write("## Plotting the decomposition in plotly")
st.plotly_chart(px.line(x=data["Date"], y=decomposition.trend, title='Trend', width=700, height=400, labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='blue'))
st.plotly_chart(px.line(x=data ["Date"], y=decomposition. seasonal, title='Seasonality', width=700, height=400, labels={'x': 'Date', 'y':'Price'}).update_traces(line_color='green'))
st.plotly_chart(px.line (x=data["Date"], y=decomposition. resid, title= 'Residuals', width=700, height=400, labels={'x': 'Date', 'y':'Price'}).update_traces(line_color='Red',line_dash='dot'))

#lets Run the model
#user input for three parameter of the model and seasonal order
p=st.slider("select the value of p",0,5,2)
d=st.slider("select the value of d",0,5,1)
q=st.slider("select the value of q",0,5,2)
seasonal_oder=st.number_input("Select the value of seasonal p",0,24,12)

model=sm.tsa.statespace.SARIMAX(data[column],order=(p,d,q),seasonal_order=(p,d,q,seasonal_oder))
model=model.fit()

#print model summary
st.header("Model Summary")
st.write(model.summary())
st.write("---")

#predict the values with user input values
st.write("<p style='color:green; font-size: 50px; font-weight: bold;'>Forecasting the data</p>", unsafe_allow_html=True)
forecast_period = st.number_input("## Enter forecast period in days", value=16)

#predict the future value(forecasting)
#forecast_period=st.number_input("Select the number of day to forecast",1,365,10)
#predict the future value
predictions=model.get_prediction(start=len(data),end=len(data)+forecast_period-1)
predictions=predictions.predicted_mean                             
st.write(len(predictions))

# add index to results dataframe as dates
predictions.index = pd.date_range(start=end_date, periods=len (predictions), freq='D')
predictions = pd.DataFrame(predictions)
predictions.insert(0, 'Date', predictions.index) 
predictions.reset_index(drop=True, inplace=True)
st.write(" Predictions", predictions)
st.write(" Actual Data", data)
st.write("---")

#lets plot the data
fig = go.Figure()
# add actual data to the plot 
fig.add_trace(go. Scatter (x=data["Date"], y=data [column], mode='lines', name='Actual', line=dict(color='blue')))
# add predicted data to the plot
fig.add_trace(go.Scatter (x=predictions ["Date"], y=predictions ["predicted_mean"], mode='lines', name='Predicted', line=dict(color='red')))
# set the title and axis labels 
fig.update_layout(title='Actual vs Predicted', xaxis_title="Date", yaxis_title='Price', width=800, height=400)
# display the plot
st.plotly_chart(fig)

# Add button to show and hide separate plot
show_plots=False
if st.button("Show separate plot"):
    if not show_plots:
        st.write(px.line(x=data["Date"], y=data[column], title='Actual', width=700, height=400, labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='blue'))
        st.write(px.line(x=predictions["Date"], y=predictions["predicted_mean"], title='Prediction', width=700, height=400, labels={'x': 'Date', 'y':'Price'}).update_traces(line_color='green'))
        show_plots=True
    else:
        show_plots=False
#Add hide button
hide_plots=False
if st.button("Hide separate plot"):
    if not hide_plots:
        hide_plots=True
    else:
        hide_plots=False
        
    st.write("---")
        
        
st.write("<p style='color:blue ;font-weight: bold; font-size:50px;'> Arfan Anjum</p>", unsafe_allow_html=True)
#paste youtube icon from online source with link


st.write("## Connect with me on social media")
# add links to my social media # urls of the images
linkedin_url = "https://img.icons8.com/color/48/000000/linkedin.png"
github_url = "https://img.icons8.com/fluent/48/000000/github.png"
facebook_url = "https://img.icons8.com/color/48/000000/facebook-new.png"
instagram_url = "https://img.icons8.com/fluent/48/000000/instagram-new.png"

# redirect urls
linkedin_redirect_url = "https://www.linkedin.com/in/arfan-anjum-02457021a/"
github_redirect_url = "https://github.com/Anjum147" 

facebook_redirect_url="https://www.facebook.com/profile.php?id=100010129122123&mibextid=ZbWKwL"
instagram_redirect_url="https://instagram.com/arfan_anjum_149?utm_source=qr&igshid=OGIxMTE0OTdkZA=="


 #add links to the images

st.markdown (f'<a href="{github_redirect_url}"><img src="{github_url}" width="60" height="60"></a>'
              f'<a href="{linkedin_redirect_url}"><img src="{linkedin_url}" width="60" height="60"></a>'
              f'<a href="{instagram_redirect_url}"><img src="{instagram_url}" width="60" height="60"></a>'
              f'<a href="{facebook_redirect_url}"><img src="{facebook_url}" width="60" height="60"></a>', unsafe_allow_html=True)
