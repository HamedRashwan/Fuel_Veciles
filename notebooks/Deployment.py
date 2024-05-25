import streamlit as st
import pandas as pd 
import joblib 

model = joblib.load('notebooks\feul_model.h5')
scaler = joblib.load('notebooks\feulscaler.h5')
print('OKAy >>>>>>>>>>>>')