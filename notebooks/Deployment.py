import streamlit as st
import pandas as pd 
import joblib 

model=joblib.load(r'C:\Users\Hamed\Downloads\Fuel_Veciles\notebooks\feul_model.h5')
scaler=joblib.load(r'C:\Users\Hamed\Downloads\Fuel_Veciles\notebooks\feulscaler.h5')
# Title of the web app
st.title('Vehicle Performance Prediction')

# Input features
st.header('Input Features')
# Define a function to handle input conversion
def convert_to_float(value):
    try:
        return float(value)
    except ValueError:
        return 0.0

# Define inputs for all features
year = st.number_input('Year', min_value=1980, max_value=2024, step=1)
engine_index = st.number_input('Engine Index', min_value=0)
engine_cylinders = st.number_input('Engine Cylinders', min_value=1, max_value=16, step=1)
engine_displacement = st.number_input('Engine Displacement (liters)', min_value=0.0)
city_mpg_ft1 = st.number_input('City MPG FT1', min_value=0.0)
unrounded_city_mpg_ft1 = st.number_input('Unrounded City MPG FT1', min_value=0.0)
city_mpg_ft2 = st.number_input('City MPG FT2', min_value=0.0)
unrounded_city_mpg_ft2 = st.number_input('Unrounded City MPG FT2', min_value=0.0)
city_gasoline_consumption_cd = st.number_input('City Gasoline Consumption CD', min_value=0.0)
city_electricity_consumption = st.number_input('City Electricity Consumption', min_value=0.0)
city_utility_factor = st.number_input('City Utility Factor', min_value=0.0)
highway_mpg_ft1 = st.number_input('Highway MPG FT1', min_value=0.0)
unrounded_highway_mpg_ft1 = st.number_input('Unrounded Highway MPG FT1', min_value=0.0)
highway_mpg_ft2 = st.number_input('Highway MPG FT2', min_value=0.0)
unrounded_highway_mpg_ft2 = st.number_input('Unrounded Highway MPG FT2', min_value=0.0)
highway_gasoline_consumption_cd = st.number_input('Highway Gasoline Consumption CD', min_value=0.0)
highway_electricity_consumption = st.number_input('Highway Electricity Consumption', min_value=0.0)
highway_utility_factor = st.number_input('Highway Utility Factor', min_value=0.0)
unadjusted_city_mpg_ft1 = st.number_input('Unadjusted City MPG FT1', min_value=0.0)
unadjusted_highway_mpg_ft1 = st.number_input('Unadjusted Highway MPG FT1', min_value=0.0)
unadjusted_city_mpg_ft2 = st.number_input('Unadjusted City MPG FT2', min_value=0.0)
unadjusted_highway_mpg_ft2 = st.number_input('Unadjusted Highway MPG FT2', min_value=0.0)
combined_mpg_ft1 = st.number_input('Combined MPG FT1', min_value=0.0)
unrounded_combined_mpg_ft1 = st.number_input('Unrounded Combined MPG FT1', min_value=0.0)
combined_mpg_ft2 = st.number_input('Combined MPG FT2', min_value=0.0)
unrounded_combined_mpg_ft2 = st.number_input('Unrounded Combined MPG FT2', min_value=0.0)
combined_electricity_consumption = st.number_input('Combined Electricity Consumption', min_value=0.0)
combined_gasoline_consumption_cd = st.number_input('Combined Gasoline Consumption CD', min_value=0.0)
combined_utility_factor = st.number_input('Combined Utility Factor', min_value=0.0)
annual_fuel_cost_ft1 = st.number_input('Annual Fuel Cost FT1', min_value=0.0)
annual_fuel_cost_ft2 = st.number_input('Annual Fuel Cost FT2', min_value=0.0)
save_or_spend_5_year = st.number_input('Save or Spend 5 Year', min_value=0.0)
annual_consumption_in_barrels_ft1 = st.number_input('Annual Consumption in Barrels FT1', min_value=0.0)
annual_consumption_in_barrels_ft2 = st.number_input('Annual Consumption in Barrels FT2', min_value=0.0)
tailpipe_co2_ft1 = st.number_input('Tailpipe CO2 FT1', min_value=0.0)
tailpipe_co2_in_grams_mile_ft1 = st.number_input('Tailpipe CO2 in Grams/Mile FT1', min_value=0.0)
tailpipe_co2_ft2 = st.number_input('Tailpipe CO2 FT2', min_value=0.0)
tailpipe_co2_in_grams_mile_ft2 = st.number_input('Tailpipe CO2 in Grams/Mile FT2', min_value=0.0)
ghg_score = st.number_input('GHG Score', min_value=0.0)
ghg_score_alt_fuel = st.number_input('GHG Score Alt Fuel', min_value=0.0)
x2d_passenger_volume = st.number_input('2D Passenger Volume', min_value=0.0)
x2d_luggage_volume = st.number_input('2D Luggage Volume', min_value=0.0)
x4d_passenger_volume = st.number_input('4D Passenger Volume', min_value=0.0)
x4d_luggage_volume = st.number_input('4D Luggage Volume', min_value=0.0)
hatchback_passenger_volume = st.number_input('Hatchback Passenger Volume', min_value=0.0)
hatchback_luggage_volume = st.number_input('Hatchback Luggage Volume', min_value=0.0)
hours_to_charge_120v = st.number_input('Hours to Charge 120V', min_value=0.0)
hours_to_charge_240v = st.number_input('Hours to Charge 240V', min_value=0.0)
hours_to_charge_ac_240v = st.number_input('Hours to Charge AC 240V', min_value=0.0)
composite_city_mpg = st.number_input('Composite City MPG', min_value=0.0)
composite_highway_mpg = st.number_input('Composite Highway MPG', min_value=0.0)
composite_combined_mpg = st.number_input('Composite Combined MPG', min_value=0.0)
range_ft1 = st.number_input('Range FT1', min_value=0.0)
city_range_ft1 = st.number_input('City Range FT1', min_value=0.0)
highway_range_ft1 = st.number_input('Highway Range FT1', min_value=0.0)
city_range_ft2 = st.number_input('City Range FT2', min_value=0.0)
highway_range_ft2 = st.number_input('Highway Range FT2', min_value=0.0)

# Gather input into a dataframe
input_data = {
    'year': [year],
    'engine_index': [engine_index],
    'engine_cylinders': [engine_cylinders],
    'engine_displacement': [engine_displacement],
    'city_mpg_ft1': [city_mpg_ft1],
    'unrounded_city_mpg_ft1': [unrounded_city_mpg_ft1],
    'city_mpg_ft2': [city_mpg_ft2],
    'unrounded_city_mpg_ft2': [unrounded_city_mpg_ft2],
    'city_gasoline_consumption_cd': [city_gasoline_consumption_cd],
    'city_electricity_consumption': [city_electricity_consumption],
    'city_utility_factor': [city_utility_factor],
    'highway_mpg_ft1': [highway_mpg_ft1],
    'unrounded_highway_mpg_ft1': [unrounded_highway_mpg_ft1],
    'highway_mpg_ft2': [highway_mpg_ft2],
    'unrounded_highway_mpg_ft2': [unrounded_highway_mpg_ft2],
    'highway_gasoline_consumption_cd': [highway_gasoline_consumption_cd],
    'highway_electricity_consumption': [highway_electricity_consumption],
    'highway_utility_factor': [highway_utility_factor],
    'unadjusted_city_mpg_ft1': [unadjusted_city_mpg_ft1],
    'unadjusted_highway_mpg_ft1': [unadjusted_highway_mpg_ft1],
    'unadjusted_city_mpg_ft2': [unadjusted_city_mpg_ft2],
    'unadjusted_highway_mpg_ft2': [unadjusted_highway_mpg_ft2],
    'combined_mpg_ft1': [combined_mpg_ft1],
    'unrounded_combined_mpg_ft1': [unrounded_combined_mpg_ft1],
    'combined_mpg_ft2': [combined_mpg_ft2],
    'unrounded_combined_mpg_ft2': [unrounded_combined_mpg_ft2],
    'combined_electricity_consumption': [combined_electricity_consumption],
    'combined_gasoline_consumption_cd': [combined_gasoline_consumption_cd],
    'combined_utility_factor': [combined_utility_factor],
    'annual_fuel_cost_ft1': [annual_fuel_cost_ft1],
    'annual_fuel_cost_ft2': [annual_fuel_cost_ft2],
    'save_or_spend_5_year': [save_or_spend_5_year],
    'annual_consumption_in_barrels_ft1': [annual_consumption_in_barrels_ft1],
    'annual_consumption_in_barrels_ft2': [annual_consumption_in_barrels_ft2],
    'tailpipe_co2_ft1': [tailpipe_co2_ft1],
    'tailpipe_co2_in_grams_mile_ft1': [tailpipe_co2_in_grams_mile_ft1],
    'tailpipe_co2_ft2': [tailpipe_co2_ft2],
    'tailpipe_co2_in_grams_mile_ft2': [tailpipe_co2_in_grams_mile_ft2],
    'ghg_score': [ghg_score],
    'ghg_score_alt_fuel': [ghg_score_alt_fuel],
    'x2d_passenger_volume': [x2d_passenger_volume],
    'x2d_luggage_volume': [x2d_luggage_volume],
    'x4d_passenger_volume': [x4d_passenger_volume],
    'x4d_luggage_volume': [x4d_luggage_volume],
    'hatchback_passenger_volume': [hatchback_passenger_volume],
    'hatchback_luggage_volume': [hatchback_luggage_volume],
    'hours_to_charge_120v': [hours_to_charge_120v],
    'hours_to_charge_240v': [hours_to_charge_240v],
    'hours_to_charge_ac_240v': [hours_to_charge_ac_240v],
    'composite_city_mpg': [composite_city_mpg],
    'composite_highway_mpg': [composite_highway_mpg],
    'composite_combined_mpg': [composite_combined_mpg],
    'range_ft1': [range_ft1],
    'city_range_ft1': [city_range_ft1],
    'highway_range_ft1': [highway_range_ft1],
    'city_range_ft2': [city_range_ft2],
    'highway_range_ft2': [highway_range_ft2]
}

input_df = pd.DataFrame(input_data)
# Scale the input data
scaled_input = scaler.transform(input_df)
# Make prediction
prediction = model.predict(scaled_input)
# Display the prediction
st.header('Prediction')
st.write(prediction)