import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

## load the trained model and the pickle files
with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)

with open('one_hot_encoder.pkl','rb') as file:
    one_hot_encoder  = pickle.load(file)

with open('scalar.pkl','rb') as file:
    scaler = pickle.load(file)

model = load_model('model.h5')

## streamlit app
st.title('Customer churn Prediction')

# User input
geography = st.selectbox('Geography',one_hot_encoder.categories_[0])
gender = st.selectbox('Gender',label_encoder_gender.classes_)
age = st.slider('Age',18,92)
balance = st.number_input("Balance")
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure',0,10)
num_of_products = st.slider('Number of Products',1,4)
has_cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_member = st.selectbox('Is Active Member',[0,1])


input_df = pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[label_encoder_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary],
    
   
    
    
})

one_hot_geo = one_hot_encoder.transform([[geography]]).toarray()
df_geo = pd.DataFrame(
    one_hot_geo,
    columns=one_hot_encoder.get_feature_names_out(['Geography'])
)

input_df = pd.concat([input_df.reset_index(drop=True),df_geo],axis=1)

input_scale = scaler.transform(input_df)

## Predict
prediction_proba = model.predict(input_scale)[0][0]

if(prediction_proba>0.5):
    st.write("The customer is likely to churn")

else:
    st.write("The customer is not likely to churn")


st.write(prediction_proba)