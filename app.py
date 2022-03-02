
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as pdr
import datetime as dt
from keras.models import load_model
import streamlit as st



end=dt.datetime.now()
start=end-dt.timedelta(days=5000)

st.title('Stock Market Analysis','TSLA')
input=st.text_input('Stock Name')

df=pdr.get_data_yahoo(input,start,end)

st.subheader('5000 days Data')
st.write(df.describe())

st.subheader('Closing price vs Time')
fig1=plt.figure(figsize=(12,8))
plt.plot(df.Close)
st.pyplot(fig1)


st.subheader('Closing price vs Time with 100 MA and 200 MA')
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig=plt.figure(figsize=(12,8))
plt.plot(ma100,'r')
plt.plot(ma200,'g')
plt.plot(df.Close,'b')
st.pyplot(fig)



data_train=pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_test=pd.DataFrame(df['Close'][int(len(df)*0.70) : int(len(df))])


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))

data_train_array=scaler.fit_transform(data_train)





model=load_model('model.h5')
past_100_days=data_train.tail(100)
final_df=past_100_days.append(data_test,ignore_index=True)
input_data=scaler.fit_transform(final_df)

x_test=[]
y_test=[]
for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])
x_test,y_test=np.array(x_test),np.array(y_test)
y_predict=model.predict(x_test)
scaler.scale_
scale_factor=1/0.00682769
y_predict=y_predict*scale_factor
y_test=y_test*scale_factor


st.subheader('Predictions Vs Actual')
fig2=plt.figure(figsize=(12,8))
plt.plot(y_test,'b',label='Original Price')
plt.plot(y_predict,'r',label='Predicted Prcie')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)



