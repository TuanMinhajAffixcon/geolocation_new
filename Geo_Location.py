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
import os

# Set up Streamlit app

st.set_page_config(page_title='Geo Segmentation',page_icon=':earth_asia:',layout='wide')
custom_css = """
<style>
body {
    background-color: #0E1117; 
    secondary-background {
    background-color: #262730; 
    padding: 10px; 
}
</style>
"""
st.write(custom_css, unsafe_allow_html=True)
st.markdown(custom_css, unsafe_allow_html=True)

st.title("Radius Search with Selected Date Range")

# Function to calculate Haversine distance between two coordinates
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of the Earth in kilometers

    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Calculate differences
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    # Distance in kilometers
    distance = R * c
    return distance

# Function to decode geohash into latitude and longitude
def decode_geohash(geohash):
    if pd.notna(geohash):
        try:
            latitude, longitude = geohash2.decode(geohash)
            return pd.Series({'Home_latitude': latitude, 'Home_longitude': longitude})
        except ValueError:
            # Handle invalid geohashes, you can modify this part based on your requirements
            return pd.Series({'Home_latitude': None, 'Home_longitude': None})
        except TypeError:
            # Handle the case where geohash2.decode returns a single tuple
            return pd.Series({'Home_latitude': geohash[0], 'Home_longitude': geohash[1]})
    else:
        # Handle null values
        return pd.Series({'Home_latitude': None, 'Home_longitude': None})
    
# Function to generate points along the circumference of a circle
def generate_circle_points(center_lat, center_lon, radius, num_points=100):
    circle_points = []
    for i in range(num_points):
        angle = 2 * radians(i * (360 / num_points))
        lat = center_lat + (radius / 111.32) * sin(angle)
        lon = center_lon + (radius / (111.32 * cos(center_lat))) * cos(angle)
        circle_points.append((lat, lon))
    return circle_points
# Generate random datetime values
start_date = pd.Timestamp('2023-12-01')
end_date = pd.Timestamp('2023-12-31')

# # Allow user to pick start and end dates
selected_start_date = st.sidebar.date_input("Select Start Date", start_date)
selected_end_date = st.sidebar.date_input("Select End Date", end_date)

dist = st.radio("Select Distance Unit", ["Meters","Kilometers"])

# # Convert date inputs to datetime objects
selected_start_date = pd.to_datetime(selected_start_date)
selected_end_date = pd.to_datetime(selected_end_date)

df = pd.read_csv('10000_Movements.csv', sep=",").dropna(subset=['latitude', 'longitude'])
df['datetimestamp'] = pd.to_datetime(df['datetimestamp'])

df = df[(df['datetimestamp'] >= selected_start_date) & (df['datetimestamp'] <= selected_end_date)]


st.text(f"Number of records within Date Range: {len(df)}")

# User input for specific locations in Australia
user_input_lat = st.sidebar.text_input("Enter a User latitude:", value="-33.8833")
user_input_lon = st.sidebar.text_input("Enter a User longitude :", value="151.2050")



if dist == 'Kilometers':
    radius_input = st.slider("Select radius (in kilometers):", min_value=1, max_value=100, value=10)

elif dist == 'Meters':
    radius_input = st.slider("Select radius (in Meters):", min_value=1, max_value=1000, value=15)
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


    # Perform radius search and count points within the specified radius
    count_within_radius = 0
    for index, row in df.iterrows():
        distance = haversine(user_lat, user_lon, row['latitude'], row['longitude'])
        if distance <= radius_input:
            count_within_radius += 1

    # Display the count
    st.text(f"Number of all devices within {radius_input} km radius: {count_within_radius}")

    # Draw a circle around the user-specified location
    circle_points = generate_circle_points(user_lat, user_lon, radius_input)
    folium.PolyLine(circle_points, color='green', weight=2.5, opacity=1).add_to(m)
    filtered_df = df[df.apply(lambda row: haversine(user_lat, user_lon, row['latitude'], row['longitude']) <= radius_input, axis=1)]
    filtered_df_maid_unique_count = filtered_df['maid'].nunique()

    if filtered_df_maid_unique_count !=0:
        filtered_df[['Home_latitude', 'Home_longitude']] = filtered_df['homegeohash9'].apply(decode_geohash)
        filtered_df[['Work_latitude', 'Work_longitude']] = filtered_df['workgeohash'].apply(decode_geohash)


        filtered_df['Distance_To_Home (Km)'] = filtered_df.apply(lambda row:
            haversine(user_lat, user_lon, float(row['Home_latitude']), float(row['Home_longitude']))
            if pd.notna(row['Home_latitude']) and pd.notna(row['Home_longitude']) else None, axis=1)
        filtered_df['Distance_To_WorkPlace (Km)'] = filtered_df.apply(lambda row:
            haversine(user_lat, user_lon, float(row['Work_latitude']), float(row['Work_longitude']))
            if pd.notna(row['Work_latitude']) and pd.notna(row['Work_longitude']) else None, axis=1)


        filtered_df_homegeo_unique_count = filtered_df[filtered_df['Distance_To_Home (Km)']<=radius_input]['homegeohash9'].nunique()
        feature_group_home = folium.FeatureGroup(name='Home Locations')
        feature_group_work = folium.FeatureGroup(name='Work Locations')
        for lat, lon in zip(filtered_df[filtered_df['Distance_To_Home (Km)']<=radius_input]['latitude'], filtered_df[filtered_df['Distance_To_Home (Km)']<=radius_input]['longitude']):
            color = 'Orange'
            folium.CircleMarker(location=[lat, lon], radius=2, color=color, fill=True, fill_color=color,
                                fill_opacity=1).add_to(feature_group_home)


        filtered_df_workgeo_unique_count = filtered_df[filtered_df['Distance_To_WorkPlace (Km)']<=radius_input]['workgeohash'].nunique()
        for lat, lon in zip(filtered_df[filtered_df['Distance_To_WorkPlace (Km)']<=radius_input]['latitude'], filtered_df[filtered_df['Distance_To_WorkPlace (Km)']<=radius_input]['longitude']):
            color = 'black'
            folium.CircleMarker(location=[lat, lon], radius=2, color=color, fill=True, fill_color=color,
                                fill_opacity=1).add_to(feature_group_work)
        filtered_df_workgeo_and_home_unique_count = filtered_df[(filtered_df['Distance_To_WorkPlace (Km)']<=radius_input) & (filtered_df['Distance_To_Home (Km)']<=radius_input)]['workgeohash'].nunique()

        # Add feature groups to the map
        feature_group_home.add_to(m)
        feature_group_work.add_to(m)

        # Add legend to the map
        folium.LayerControl().add_to(m)

        same_values_count = (filtered_df['homegeohash9'] == filtered_df['workgeohash']).value_counts()
        sum_of_true = same_values_count.get(True, 0)

        # if sum_of_true == 0:
        #     st.write("All are unique")

        st.text(f"Number of Unique device ids within {radius_input} km radius: {filtered_df_maid_unique_count}")
        st.text(f"Total Home counts within {radius_input} km radius {filtered_df_homegeo_unique_count}")
        st.text(f"Total WorkPlace counts within {radius_input} km radius: {filtered_df_workgeo_unique_count}")
        st.text(f"Total WorkPlace and Home counts within {radius_input} km radius: {filtered_df_workgeo_and_home_unique_count}")



        day_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']

        # col1=st.columns((1))
        # with col1:
        folium_static(m)
        fig1 = px.histogram(filtered_df, x=filtered_df['datetimestamp'].dt.hour, nbins=24, labels={'datetimestamp': 'Hour of Day', 'count': 'Count'})
        filtered_df['day_of_week'] = filtered_df['datetimestamp'].dt.dayofweek.map(lambda x: day_names[x])
        fig2 = px.histogram(filtered_df, x=filtered_df['day_of_week'], nbins=7,
                    labels={'day_of_week': 'Day of the Week', 'count': 'Count'},
                    category_orders={'day_of_week': day_names})
        fig1.update_traces(marker_color='yellow', opacity=0.7)

        # Set background color to be transparent
        fig1.update_layout({
            'plot_bgcolor': 'rgba(0, 0, 0, 0)',
            'paper_bgcolor': 'rgba(0, 0, 0, 0)',
            'xaxis': {'showgrid': False,'title': 'Hour'},
            'yaxis': {'showgrid': False,'title': 'Total Count'},
        })
        fig2.update_layout({
            'plot_bgcolor': 'rgba(0, 0, 0, 0)',
            'paper_bgcolor': 'rgba(0, 0, 0, 0)',
            'xaxis': {'showgrid': False,'title': 'Days'},
            'yaxis': {'showgrid': False,'title': 'Total Count'},
        })

        # with col1:
        st.write("Histogram of Hour Variations")
        st.plotly_chart(fig1)
        st.write("Histogram of Day Variations")
        st.plotly_chart(fig2)

    else:
        st.warning("No Records Founds")

else:
    st.warning("Please enter both latitude and longitude values.")


