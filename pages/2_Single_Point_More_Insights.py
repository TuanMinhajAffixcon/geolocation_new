from pyspark.sql import SparkSession
from pyspark.sql.functions import year, month, dayofmonth, dayofweek, hour,col, desc, count,when
from pyspark.sql import functions as F
import pandas as pd
import streamlit as st
from streamlit_folium import folium_static
import folium
import plotly.express as px

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

st.title("Additional Insights")
spark = SparkSession.builder \
    .appName("YourAppName") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

df = spark.read.csv("10000_Movements.csv", header=True, inferSchema=True)
df.select(['maid','datetimestamp','latitude','longitude','workgeohash','homegeohash9']).show(5)
df.printSchema()
df = df.withColumn('year', year('datetimestamp')) \
       .withColumn('month', month('datetimestamp')) \
       .withColumn('day', dayofmonth('datetimestamp')) \
       .withColumn('day_of_week', dayofweek('datetimestamp')) \
       .withColumn('hour', hour('datetimestamp'))

df.select(['maid', 'year', 'month', 'day', 'day_of_week', 'hour', 'latitude', 'longitude', 'workgeohash', 'homegeohash9']).show(5)
day_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
df = df.withColumn('day_name', when(df.day_of_week == 1, 'Sunday')
                                .when(df.day_of_week == 2, 'Monday')
                                .when(df.day_of_week == 3, 'Tuesday')
                                .when(df.day_of_week == 4, 'Wednesday')
                                .when(df.day_of_week == 5, 'Thursday')
                                .when(df.day_of_week == 6, 'Friday')
                                .when(df.day_of_week == 7, 'Saturday'))

df = df.withColumn('day_type', 
                   when((df.day_of_week <= 5) & (df.day_of_week > 1), 'Weekday')
                   .otherwise('Weekend'))
df.select(['maid', 'year', 'month', 'day', 'day_name', 'day_type', 'hour', 'latitude', 'longitude', 'workgeohash', 'homegeohash9']).show(5)


start_date = pd.Timestamp('2023-12-01')
end_date = pd.Timestamp('2023-12-31')

selected_start_date = st.sidebar.date_input("Select Start Date", start_date)
selected_end_date = st.sidebar.date_input("Select End Date", end_date)

dist = st.radio("Select Distance Unit", ["Meters","Kilometers"])
selected_start_date = pd.to_datetime(selected_start_date)
selected_end_date = pd.to_datetime(selected_end_date)
filtered_df = df[(df['datetimestamp'] >= selected_start_date) & (df['datetimestamp'] <= selected_end_date)]

user_input_lat = st.sidebar.text_input("Enter a User latitude:", value="-37.82968153089708")
user_input_lon = st.sidebar.text_input("Enter a User longitude :", value="145.05531534492368")


center = (-37.82968153089708, 145.05531534492368)
if user_input_lat =='-37.82968153089708' and user_input_lon=='145.05531534492368':
    st.sidebar.text("Dan Murphy's Camberwell")

if dist == 'Kilometers':
    radius_input = st.slider("Select radius (in kilometers):", min_value=1, max_value=100, value=10)

elif dist == 'Meters':
    radius_input = st.slider("Select radius (in Meters):", min_value=1, max_value=1000, value=15)
    radius_input=radius_input/1000

user_lat = float(user_input_lat)
user_lon = float(user_input_lon)

distance = F.acos(
    F.sin(F.radians(F.lit(user_lat))) * F.sin(F.radians(filtered_df['latitude'])) +
    F.cos(F.radians(F.lit(user_lat))) * F.cos(F.radians(filtered_df['latitude'])) *
    F.cos(F.radians(filtered_df['longitude']) - F.radians(F.lit(user_lon)))
) * F.lit(6371.0)  # Earth radius in kilometers

# Filter the DataFrame based on the distance condition
filtered_df = filtered_df.withColumn('distance', distance)
count_within_radius = filtered_df.filter(filtered_df['distance'] <= radius_input).count()
count_within_radius_df = filtered_df.filter(filtered_df['distance'] <= radius_input)

st.write(count_within_radius)

coordinates = filtered_df.select('latitude', 'longitude').collect()

mymap = folium.Map(location=[user_lat, user_lon], zoom_start=10)

for coord in coordinates:
    folium.CircleMarker(location=[coord['latitude'], coord['longitude']], radius=2, color='blue', fill=True, fill_color='blue',
                            fill_opacity=1).add_to(mymap)
folium.CircleMarker(location=[user_lat, user_lon], radius=4, color='red', fill=True, fill_color='red',
                        fill_opacity=1).add_to(mymap)
folium.Circle(
    location=(user_lat,user_lon),
    radius=radius_input*1000,
    color='green',
    fill=True,
    fill_opacity=0.4,
    ).add_to(mymap)
col1,col2=st.columns((0.6,0.3))
with col1:
    folium_static(mymap)
with col2:
    with st.expander('About',expanded=True):
        st.write('This Map is plotting all the records for seleted date range with user radius')






st.markdown("Frequent Visition")
count_within_radius_df_maid=count_within_radius_df.groupBy(['maid']).agg(count('*').alias('count')).orderBy(desc('count'))
# count_within_radius_df_maid=count_within_radius_df_maid.toPandas()
freq_day=count_within_radius_df.groupBy(['maid','day']).agg(count('*').alias('count')).orderBy(('count'))
with st.expander('View grouped maids'):
    col1,col2=st.columns((2))
    with col1:
        st.write('All days freuency',count_within_radius_df_maid)
    with col2:
        st.write('Days wise freuency',freq_day)

pandas_df = count_within_radius_df.toPandas()

pandas_unique = pandas_df.drop_duplicates(subset=['maid'])

day_pattern_fig = px.bar(pandas_df.groupby(['day']).size().reset_index(name='count'), 
                            x='day', y='count', color='day',
                            title='Daily Pattern', labels={'count': 'Count'}, text='count')
day_pattern_fig.update_layout(xaxis=dict(showgrid=False),
                                  yaxis=dict(showgrid=False))

hourly_pattern_fig = px.bar(pandas_df.groupby(['hour']).size().reset_index(name='count'), 
                            x='hour', y='count', color='hour',
                            title='Hourly Pattern', labels={'count': 'Count'}, text='count')
hourly_pattern_fig.update_layout(xaxis=dict(showgrid=False),
                                  yaxis=dict(showgrid=False))
age_pattern_fig = px.bar(pandas_unique.groupby(['Age_Range']).size().reset_index(name='count'), 
                            x='Age_Range', y='count', color='count',
                            title='Age_Range Variation', labels={'count': 'Count'}, text='count')
age_pattern_fig.update_layout(xaxis=dict(showgrid=False),
                                  yaxis=dict(showgrid=False))
gender_pattern_fig = px.pie(pandas_unique, names='Gender', title='Gender Variation')

day_pattern_fig.update_layout(barmode='group')

colors = {'Monday': 'rgb(31, 119, 180)', 'Tuesday': 'rgb(31, 119, 180)', 'Wednesday': 'rgb(31, 119, 180)',
          'Thursday': 'rgb(31, 119, 180)', 'Friday': 'rgb(31, 119, 180)',
          'Saturday': 'rgb(255, 127, 14)', 'Sunday': 'rgb(255, 127, 14)'}

pandas_df['day_name'] = pd.Categorical(pandas_df['day_name'], categories=list(colors.keys()), ordered=True)
weekday_weekend_fig = px.bar(pandas_df.groupby('day_name').size().reset_index(name='count'), 
                             x='day_name', y='count', color='day_name',
                             color_discrete_map=colors, title='Weekday Vs Weekend Pattern',
                             labels={'count': 'Percentage'}, text='count')
weekday_weekend_fig.update_layout(xaxis=dict(showgrid=False),
                                  yaxis=dict(showgrid=False))

stack_age_pattern_fig = px.bar(pandas_unique.groupby(['Age_Range', 'Gender']).size().reset_index(name='Count'), x='Age_Range', y='Count', color='Gender', 
                         title='Age Range Variation by Gender', labels={'Count': 'Count'})
stack_age_pattern_fig.update_layout(xaxis=dict(showgrid=False),
                                  yaxis=dict(showgrid=False))
weekday_weekend_fig.update_layout(barmode='group')

st.plotly_chart(day_pattern_fig)
st.plotly_chart(hourly_pattern_fig)
st.plotly_chart(weekday_weekend_fig)
st.plotly_chart(age_pattern_fig)
st.plotly_chart(gender_pattern_fig)
st.plotly_chart(stack_age_pattern_fig)


heatmap_data = pandas_df.pivot_table(index='day_name', columns='hour', aggfunc='size')

# Reorder the rows of the heatmap data to match the order in the image
heatmap_data = heatmap_data.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

import plotly.graph_objects as go

# Convert the heatmap_data DataFrame to a list of lists
heatmap_values = heatmap_data.values.tolist()

# Create the heatmap using Plotly Graph Objects
fig = go.Figure(data=go.Heatmap(
                   z=heatmap_values,
                   x=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24'],
                   y=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                   colorscale='YlOrRd'))

# Update layout
fig.update_layout(title='Visitation Insights',
                  xaxis_title='Hour of the Day',
                  yaxis_title='Day of the Week')

# Show the plot
st.plotly_chart(fig)

from pyspark.sql.functions import unix_timestamp, col

hour_difference_df = count_within_radius_df.groupBy('maid', 'day').agg(
    (F.max(unix_timestamp('datetimestamp')) - F.min(unix_timestamp('datetimestamp'))).alias('hour_difference')
)
hour_difference_df = hour_difference_df.withColumn('hour_difference', col('hour_difference') / 3600) # Convert seconds to hours

hour_difference_df=hour_difference_df.where(hour_difference_df['hour_difference'] > 0).orderBy(('hour_difference'))

hour_difference_pandas = hour_difference_df.select('hour_difference').toPandas()

# Define bin edges
bin_edges = [0, 2, 5, 24]

# Categorize minute differences into bins
hour_difference_pandas['bin'] = pd.cut(hour_difference_pandas['hour_difference'], bins=bin_edges, labels=['0-2', '2-5', '5-24'])

# Aggregate counts by bin
agg_df = hour_difference_pandas.groupby('bin').size().reset_index(name='count')

# Create histogram using Plotly Express
histogram_fig = px.bar(agg_df, x='bin', y='count', 
                       title='Histogram of Hour Spending',
                       labels={'bin': 'Hour Spending Range', 'count': 'Frequency'})
histogram_fig.update_layout(xaxis=dict(showgrid=False),
                                  yaxis=dict(showgrid=False))

col1,col2=st.columns((0.6,0.3))
with col1:
    st.plotly_chart(histogram_fig)
with col2:
    st.write(hour_difference_df)


count_within_radius_df_filtered = df.join(count_within_radius_df, on='maid', how='left_semi')
lon_lat=count_within_radius_df_filtered.select('latitude','longitude').orderBy('maid')
lon_lat=lon_lat.toPandas()
lon_lat = lon_lat.sample(frac=1).reset_index(drop=True)  # Shuffle the DataFrame for better visualization
# Create a scatter plot using Plotly Express
fig = px.scatter(lon_lat, x='longitude', y='latitude', title='Plot of Latitude and Longitude who visited in given area')
fig.update_traces(marker=dict(size=8, opacity=0.5, color='blue'))
fig.update_layout(xaxis_title='Longitude', yaxis_title='Latitude', plot_bgcolor='rgba(0,0,0,0)')
st.plotly_chart(fig)

from math import radians, sin, cos, sqrt, atan2

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
    distance = R * c*1000
    return distance

# Collect latitude and longitude values from the DataFrame
coordinates = count_within_radius_df_filtered.select('latitude', 'longitude').collect()

# Convert list of Row objects to Pandas DataFrame
coordinates_df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])

# Create a Folium map centered at the first coordinate
mymap_filtered = folium.Map(location=[user_lat, user_lon], zoom_start=10)

# Define he radius of the circle
# circle_radius = 5000  # in meters
# Add markers for all coordinates to the map
for _, row in coordinates_df.iterrows():
    # Calculate the distance from the user's location to the current point
    distance = haversine(user_lat, user_lon, row['latitude'], row['longitude'])
    if distance <= radius_input*1000:
        # Add marker inside the circle with blue color
        folium.CircleMarker(location=[row['latitude'], row['longitude']], radius=2, color='blue', fill=True, fill_color='blue',
                            fill_opacity=1).add_to(mymap_filtered)
    else:
        # Add marker outside the circle with orange color
        folium.CircleMarker(location=[row['latitude'], row['longitude']], radius=2, color='brown', fill=True, fill_color='brown',fill_opacity=1).add_to(mymap_filtered)

# Add the user's location marker
folium.Marker(location=[user_lat, user_lon], radius=4, color='red', fill=True, fill_color='red',
                    fill_opacity=1).add_to(mymap_filtered)

# Add the circle representing the radius
folium.Circle(location=(user_lat, user_lon), radius=radius_input*1000, color='green', fill=True, fill_opacity=0.4).add_to(mymap_filtered)

col1,col2=st.columns((0.6,0.3))
with col1:
    folium_static(mymap_filtered)
with col2:
    with st.expander('About',expanded=True):
        st.write('This Map is plotting who visited the place and their other visiting points in seleted date range with user radius')