import streamlit as st
import pandas as pd 
import joblib 

model=joblib.load('feul_model.h5')
scaler=joblib.load('feulscaler.h5')

st.title('Feul Veclies Streamlit App')
st.info('just building a testing app for out ml model ')

col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col13=st.columns(13)

col1.metric('year','2017')
col2.metric('engine_cylinders','2345')
col3.metric('engine_displacement','2345')
col4.metric('combined_mpg_ft1','2345')
col5.metric('combined_mpg_ft2','2345')
col6.metric('tailpipe_co2_in_grams_mile_ft1','2345')
col7.metric('tailpipe_co2_ft1','2345')
col8.metric('tailpipe_co2_in_grams_mile_ft2','2345')
col9.metric('tailpipe_co2_ft2','2345')
col10.metric('unadjusted_highway_mpg_ft1','2345')
col11.metric('city_gasoline_consumption_cd','2345')
col12.metric('city_range_ft2','2345')
col13.metric('fuel_economy_score','2345')

year=st.number_input('Enter Year: ')
engine_cylinders=st.slider('Engine_Cylinders? ',0,16,8)
engine_displacement=st.number_input('Enter Displacement: ')
combined_mpg_ft1=st.number_input('Enter Combined_MPG_ft1: ')
combined_mpg_ft2=st.number_input('Enter Combined_MPG_ft2: ')
tailpipe_co2_in_grams_mile_ft1=st.number_input('Enter Tailpipe_CO2_In_Grams_Mile_ft1: ')
tailpipe_co2_ft1=st.number_input('Enter Tailpipe_CO2_ft1: ')
tailpipe_co2_in_grams_mile_ft2=st.number_input('Enter Tailpipe_CO2_In_Grams_Mile_ft2: ')
tailpipe_co2_ft2=st.number_input('Enter Tailpipe_CO2_FT2: ')
unadjusted_highway_mpg_ft1=st.number_input('Enter Unadjusted_Highway_MPG_FT1: ')
city_gasoline_consumption_cd=st.number_input('Enter City_Gasoline_Consumption_CD: ')
city_range_ft2=st.number_input('Enter City_Range_FT2: ')
fuel_economy_score=st.number_input('Enter Fuel_Economy_Score: ')

data=[year,engine_cylinders,engine_displacement,combined_mpg_ft1,combined_mpg_ft2,tailpipe_co2_in_grams_mile_ft1,tailpipe_co2_ft1,tailpipe_co2_in_grams_mile_ft2,tailpipe_co2_ft2 ,unadjusted_highway_mpg_ft1,city_gasoline_consumption_cd,city_range_ft2,fuel_economy_score]

st.write(data)

data_scaled=scaler.transform([data])
result=model.predict(data_scaled)

st.write(result)