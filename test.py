import pandas as pd
import streamlit as st
import geohash2
from math import radians, sin, cos, sqrt, atan2
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
import folium
from streamlit_folium import folium_static
import pyodbc
import os

# User input for specific locations in Australia
user_input_lat = st.sidebar.text_input("Enter a User latitude:", value="-33.8833")
user_input_lon = st.sidebar.text_input("Enter a User longitude :", value="151.2050")

dist = st.radio("Select Distance Unit", ["Meters","Kilometers"])
df = pd.read_csv('random_coordinates_with_geohashes.csv', sep=",").dropna(subset=['latitude', 'longitude'])


if dist == 'Kilometers':
    radius_input = st.slider("Select radius (in kilometers):", min_value=1, max_value=100, value=10)

elif dist == 'Meters':
    radius_input = st.slider("Select radius (in Meters):", min_value=1, max_value=1000, value=758)
    radius_input=radius_input/1000

# Process user input
if user_input_lat and user_input_lon :
    user_lat = float(user_input_lat)
    user_lon = float(user_input_lon)

    # Create a folium map centered on the user-specified location
    m = folium.Map(location=[user_lat, user_lon], zoom_start=10)

    # Plot sample data as blue points
    for lat, lon in zip(df['latitude'], df['longitude']):
        color = 'blue'
        folium.CircleMarker(location=[lat, lon], radius=2, color=color, fill=True, fill_color=color,
                            fill_opacity=1).add_to(m)

    # Highlight the user-specified location as a red point
    folium.CircleMarker(location=[user_lat, user_lon], radius=4, color='red', fill=True, fill_color='red',
                        fill_opacity=1).add_to(m)

folium_static(m)

