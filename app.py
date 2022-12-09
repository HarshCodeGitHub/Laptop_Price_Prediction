import streamlit as st
import pickle 
import numpy as np
import sklearn
import pandas as pd


data = pd.read_csv("E:\VIIT TY Sem-1\Laptop_Price_Predection\laptop_data.csv")
pipe = pickle.load(open('pipe.pkl','rb'))

df = pickle.load(open('df.pkl','rb'))


st.title("Laptop")

# Brand
company = st.selectbox('Brand',df['Company'].unique())

# type of laptop
laptop_type = st.selectbox('Type',df['TypeName'].unique())

# Ram
ram_type = st.selectbox('RAM(in GB)',[2,4,6,8,12,16,24,32,64])

# Weight
weight = st.number_input('Weight of Laptop')

# TouchScreen
touchscreen = st.selectbox('Touchscreen',['No','Yes'])

# IPS
ips = st.selectbox('IPS',['No','Yes'])

# Screen Size
screen_size = st.number_input('Screen Size')

# resolution
resolution = st.selectbox('Select Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1600','2560x1600','2560x1440','2384x1440'])

# CPU
cpu = st.selectbox('CPU',df['Cpu brand'].unique())

# HDD
hdd = st.selectbox('HDD',[0,128,512,1024])

# SDD
ssd = st.selectbox('SDD',[0,128,512,1024]) 

# GPU
gpu = st.selectbox('GPU',df['Gpu brand'].unique()) 

# OS
os = st.selectbox('OS',df['os'].unique()) 

if st.button('Predict Price'):
    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2)+(Y_res**2)) ** 0.5/screen_size
    query = np.array([company,laptop_type,ram_type,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os])
    query = query.reshape(1,12)
    prediction = str((np.exp(pipe.predict(query)[0])))
    st.subheader("The Predicted Price is : " + prediction)