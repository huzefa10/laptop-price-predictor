import streamlit as st
import pickle
import numpy as np

st.title('Laptop Price Predictor')

#Import the model
pipe= pickle.load(open('pipe.pkl','rb'))
df= pickle.load(open('df.pkl','rb'))

# brand
brand = st.selectbox('Brand',df['Company'].unique())

#type of laptop
type = st.selectbox('Type',df['TypeName'].unique())

#type of RAM
ram = st.selectbox('RAM (in GB)',df['Ram'].unique())

#Weight of the Laptop
weight = st.number_input('Enter weight of the laptop')

# Touchscreen
touchscreen = st.selectbox('Touchscreen',['Yes','No'])

# IPS
ips = st.selectbox('IPS',['Yes','No'])

# Screensize
screensize = st.number_input('Enter the screensize')

# Resolutions
resolutions = st.selectbox('Screen Resolutions',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

# CPU
cpu = st.selectbox('CPU Brand',df['Cpu Brand'].unique())

# Processing Speed
process_speed = st.number_input('Enter the Processing Speed (Between 1 to 4)')

# HDD
hdd = st.selectbox('HDD (in GB)',[0, 128, 256, 512, 1024, 2048])

# SSD
ssd = st.selectbox('SSD (in GB)',[0, 8, 128, 256, 512, 1024])

# GPU
gpu = st.selectbox('GPU',df['Gpu Brand'].unique())

# OS
os = st.selectbox('OS',df['OS'].unique())

if st.button('Predict Price'):
    # Query
    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0
    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    x_res = int(resolutions.split('x')[0])
    y_res = int(resolutions.split('x')[1])
    ppi = ((x_res**2) +(y_res**2))**0.5/float(screensize)
    query = np.array([brand, type , ram, weight, touchscreen, ips, ppi, cpu, process_speed, hdd, ssd, gpu, os])

    query = query.reshape(1,13)
    st.title('The Predicted price of the laptop is: Rs '+str(int(np.exp(pipe.predict(query))[0])))
