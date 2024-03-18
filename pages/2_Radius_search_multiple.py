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
from dotenv import load_dotenv
from collections import Counter
from itertools import combinations
load_dotenv()
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

st.title("Radius Search with Multiple Locations")

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
# selected_distance = st.radio("Select Radius Type", ["Fixed Radius", "Varying Radius"])

df = pd.read_csv('10000_Movements.csv', sep=",").dropna(subset=['latitude', 'longitude'])
df[['Home_latitude', 'Home_longitude']] = df['homegeohash9'].apply(decode_geohash)
df[['Work_latitude', 'Work_longitude']] = df['workgeohash'].apply(decode_geohash)

items = {}
no_of_locations = st.text_input("Enter No of Addresses : ",value=5)
for i in range(1, int(no_of_locations) + 1):
    location_name = f"loc{i}"
    items[location_name] = []

dist = st.radio("Select Distance Unit", ["Meters","Kilometers"])

if  no_of_locations:
    no_of_locations = int(no_of_locations)

    # if selected_distance=="Fixed Radius":
    #     count_within_radius = {}
    #     filtered_coordinates = {}
    #     industry_list = ['latitude', 'longitude']
    #     user_coordinates = []

    #     for _ in range(no_of_locations):
    #         location_data = []
    #         row = st.columns(2)
            
    #         for i, industry in enumerate(industry_list):
    #             value = row[i].number_input(f"{industry} - Location {_ + 1}", format="%.6f")
    #             location_data.append(value)
            
    #         user_coordinates.append(location_data)
    #     if len(user_coordinates[0]) ==2:
    #         for user_lat, user_lon in user_coordinates:
    #             count_within_radius[(user_lat, user_lon)] = 0
    #             filtered_coordinates[(user_lat, user_lon)] = []
    #         if dist == 'Kilometers':
    #             radius_input = st.slider("Select radius (in kilometers):", min_value=1, max_value=100, value=10)

    #         elif dist == 'Meters':
    #             radius_input = st.slider("Select radius (in Meters):", min_value=1, max_value=1000, value=10)
    #             radius_input=radius_input/1000

    #         for index, row in df.iterrows():
    #             for user_lat, user_lon in user_coordinates:
    #                 distance = haversine(user_lat, user_lon, row['latitude'], row['longitude'])
    #                 if distance <= radius_input:
    #                     count_within_radius[(user_lat, user_lon)] += 1
    #                     filtered_coordinates[(user_lat, user_lon)].append((row['latitude'], row['longitude']))
    #     else:
    #         st.warning("Please Enter Only Longitude and Latitude Sequentially")

    # else:
    count_within_radius = {}
    filtered_coordinates = {}
    home_counts = {}
    home_counts = {}
    work_counts = {}

    industry_list = ['latitude', 'longitude', 'Radius']
    user_coordinates = []

    default_values = [[-33.864201, 151.21644, 30.0], [-33.8471, 151.0634, 11.0], [-33.866159, 151.215256, 28.0], [-33.8833, 151.2050, 22.0], [-33.893000, 151.215000, 21.0]]

    for idx in range(no_of_locations):
        location_data = []
        row = st.columns(3)

        # Get the default values for the current location
        default_value = default_values[idx] if idx < len(default_values) else [0.0, 0.0, 0.0]

        for i, industry in enumerate(industry_list):
            value = row[i].number_input(f"{industry} - Location {idx + 1}", format="%.6f", value=default_value[i])
            location_data.append(value)
        
        user_coordinates.append(location_data)
    if len(user_coordinates[0]) ==3:
        for user_lat, user_lon, radius_input in user_coordinates:
            count_within_radius[(user_lat, user_lon)] = 0
            filtered_coordinates[(user_lat, user_lon)] = []
            home_counts[(user_lat, user_lon)] = 0
            work_counts[(user_lat, user_lon)] = 0

        for index, row in df.iterrows():
            for user_lat, user_lon, radius in user_coordinates:
                distance = haversine(user_lat, user_lon, row['latitude'], row['longitude'])
                home_distance = haversine(user_lat, user_lon, float(row['Home_latitude']), float(row['Home_longitude']))
                work_distance = haversine(user_lat, user_lon, float(row['Work_latitude']), float(row['Work_longitude']))



                if dist == 'Meters':
                    radius= radius/1000

                if distance <= radius:
                    count_within_radius[(user_lat, user_lon)] += 1
                    filtered_coordinates[(user_lat, user_lon)].append((row['latitude'], row['longitude']))
                if home_distance <= radius:
                    home_counts[(user_lat, user_lon)] += 1
                if work_distance <= radius:
                    work_counts[(user_lat, user_lon)] += 1
        

    else:
        st.warning("Please Enter Only Longitude and Latitude and Radius Sequentially")


    # all_radius = [sublist[2] for sublist in user_coordinates]
    for idx, i in enumerate(user_coordinates):

    # for radius in all_radius:
        if dist == 'Meters':
            i[2]= i[2]/1000
        filtered_df = df[df.apply(lambda row: haversine(i[0], i[1], row['latitude'], row['longitude']) <= i[2], axis=1)]
        if len(filtered_df) !=0:
            filtered_df[['Home_latitude', 'Home_longitude']] = filtered_df['homegeohash9'].apply(decode_geohash)
            filtered_df[['Work_latitude', 'Work_longitude']] = filtered_df['workgeohash'].apply(decode_geohash)

        filtered_df['Distance_To_Home (Km)'] = filtered_df.apply(lambda row:
            haversine(i[0], i[1], float(row['Home_latitude']), float(row['Home_longitude']))
            if pd.notna(row['Home_latitude']) and pd.notna(row['Home_longitude']) else None, axis=1)
        filtered_df['Distance_To_WorkPlace (Km)'] = filtered_df.apply(lambda row:
            haversine(i[0], i[1], float(row['Work_latitude']), float(row['Work_longitude']))
            if pd.notna(row['Work_latitude']) and pd.notna(row['Work_longitude']) else None, axis=1)

        unique_maid = filtered_df['maid'].unique()
        items[f'loc{idx + 1}'] = unique_maid

        
    count_within_radius_df = pd.DataFrame(list(count_within_radius.items()), columns=['coordinates', 'Total_record_counts'])
    home_counts_df = pd.DataFrame(list(home_counts.items()), columns=['coordinates', 'home_counts'])
    work_counts_df = pd.DataFrame(list(work_counts.items()), columns=['coordinates', 'work_counts'])
    count_df = pd.concat([count_within_radius_df, home_counts_df, work_counts_df], axis=1)


    # Display the count DataFrame
    st.write("Count within radius for each location:")
    st.write(count_df.set_index('coordinates'))
    # st.write(items)
    # Create a counter for all items
    all_items_counter = Counter([item for sublist in items.values() for item in sublist])

    # Find records that are unique to each location
    unique_records_per_location = {}
    for location, records in items.items():
        location_counter = Counter(records)
        unique_records_per_location[location] = [item for item in records if location_counter[item] == 1 and all_items_counter[item] == 1]

    locations_list = []
    common_items_list = []
    item_count_list = []
    # Print unique records for each location
    for location, unique_records in unique_records_per_location.items():
        locations_list.append(location)
        common_items_list.append(unique_records)
        item_count_list.append(len(unique_records))
    # Create the DataFrame
    df_common_items_counts = pd.DataFrame({
        'Locations': locations_list,
        'Unique Items': common_items_list,
        'Count': item_count_list
    })

    df_common_items_counts=df_common_items_counts[df_common_items_counts['Count']>0]

    # Display the DataFrame
    st.write('Unique records for each location only',df_common_items_counts)

    #-----------------------------------------------------------
    # Initialize a dictionary to store common items and their counts for each combination of locations
    common_items_counts = {}

    # Calculate common items and their counts for each combination of locations
    for combination_size in range(2, len(items) + 1):
        for combination in combinations(items.keys(), combination_size):
            common_items = set.intersection(*(set(items[loc]) for loc in combination))
            common_items_key = tuple(sorted(combination))
            if common_items_key not in common_items_counts:
                common_items_counts[common_items_key] = Counter()
            common_items_counts[common_items_key].update(common_items)
    # Initialize lists to store data
    locations_list = []
    common_items_list = []
    item_count_list = []

    # Iterate through the common_items_counts dictionary
    for combination, counts in common_items_counts.items():
        # Extract the locations, common items, and their counts
        locations = ', '.join(combination)
        common_items = ', '.join([item for item, count in counts.items() if count > 0])
        item_count = sum(counts[item] for item in counts if counts[item] > 0)
        
        # Append the data to the lists
        locations_list.append(locations)
        common_items_list.append(common_items)
        item_count_list.append(item_count)

    # Create the DataFrame
    df_common_items_counts = pd.DataFrame({
        'Locations': locations_list,
        'Common Items': common_items_list,
        'Count': item_count_list
    })

    df_common_items_counts=df_common_items_counts[df_common_items_counts['Count']>0]

    # Display the DataFrame
    # st.write('Common records for multiple locations',df_common_items_counts)



else:
    st.warning("Please Enter No of Locations")




