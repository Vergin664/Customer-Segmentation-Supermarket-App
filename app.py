
import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load the instances that where created

with open('final_model.pkl','rb') as file:
    model=pickle.load(file)


with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)

def prediction(input_data):
    scaled_data=scaler.transform(input_data)
    pred=model.predict(scaled_data)[0]

    if pred==0:
        return 'Luxury Spenders'
    elif pred==1:
        return 'Budget Enthusiasts'
    elif pred==2:
        return 'Potential Upscalers'
    else:
        return 'Practical Shoppers'

def main():
    st.title('Customer Segmentation for Supermarket Membership Analysis')
    st.subheader('Identifying customer groups based on income, age, and spending behavior to optimize marketing strategies and drive business growth.')
    Yearly_Income = st.number_input('Enter the Annual Income (in $)', min_value=0, step=1000)
    Age = st.number_input('Enter your Age', min_value=0, step=1)
    Cust_Spend_Score = st.slider('Enter the Spending Score (0-100)', min_value=0, max_value=100, step=1)

    input_list=[[Yearly_Income,Age,Cust_Spend_Score]]

    if st.button('Predict'):
        response = prediction(input_list)
        st.success(response)

if __name__ == '__main__':
    main()
