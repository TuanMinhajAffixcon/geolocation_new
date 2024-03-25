# import streamlit as st
# import folium
# import numpy as np
# from shapely.geometry import Point, Polygon
# from streamlit_folium import folium_static
# from pyathena import connect
# import boto3
# import pandas as pd

# athena_client = boto3.client('athena', region_name='ap-southeast-2')

# # # Replace 'your-access-key-id' and 'your-secret-access-key' with your AWS credentials
# conn = connect(aws_access_key_id='AKIA2ZITI36WEZ3CLDHD',
#             aws_secret_access_key='0zGYIEMTtMftlT655JUDMs9UgHUXaJBOLiyLVft5',
#             s3_staging_dir='s3://tuan-query-result-bucket/query results/',
#             region_name='ap-southeast-2')


# mycursor = conn.cursor()
# decimal_places = 3
# lower_lat = round((int(-37.82965864354029 * 10**decimal_places) / 10**decimal_places)-0.001,3)
# upper_lat = round((int(-37.82965864354029 * 10**decimal_places) / 10**decimal_places)+0.001,3)
# lower_lon = round((int(145.05527771595223 * 10**decimal_places) / 10**decimal_places)-0.001,3)
# upper_lon = round((int(145.05527771595223 * 10**decimal_places) / 10**decimal_places)+0.001,3) 

# mycursor.execute(f"SELECT maid, latitude, longitude, year, month, datetimestamp FROM lifesight.tbl_movement_geohash_parquet \
#                   WHERE (month='02' AND year='2024') AND (latitude<={upper_lat} AND latitude>={lower_lat}) AND \
#                         (longitude<={upper_lon} AND longitude>={lower_lon})")

# chunk_size = 10000
# chunks = []

# while True:
#     chunk = mycursor.fetchmany(chunk_size)
#     if not chunk:
#         break
#     chunks.append(chunk)

# column_names = [desc[0] for desc in mycursor.description]
# df_movement = pd.DataFrame([item for sublist in chunks for item in sublist], columns=column_names)
# lat_of_lists = df_movement['latitude'].values.tolist()
# lon_of_lists = df_movement['longitude'].values.tolist()
# lat_lon_list = list(zip(lat_of_lists, lon_of_lists))

# # st.write(df_movement)
# st.write(lat_lon_list[:5])

# st.write('success')

# # # # Center point coordinates
# # center_point = (-37.82967324035952, 145.05530846706537)
# # # Generate random points around the center point
# # num_points = 500
# # radius = 0.001  # Assuming 1 degree of latitude/longitude is approximately 111 km
# # angles = np.random.uniform(0, 2*np.pi, num_points)
# # distances = np.random.uniform(0, radius, num_points)  # Assuming 0.001 degree is approximately 100 meters

# # random_points = []
# # for angle, distance in zip(angles, distances):
# #     lat = center_point[0] + np.sin(angle) * distance
# #     lon = center_point[1] + np.cos(angle) * distance
# #     random_points.append((lat, lon))

# # st.write(random_points)
# # # Create a Folium map centered on the center point
# # m = folium.Map(location=[center_point[0], center_point[1]], zoom_start=15)

# # # Mark each random point with a circle
# # for point in random_points:
# #     folium.Circle(location=point, radius=0.01, color='blue', fill=True, fill_color='blue', fill_opacity=0.6).add_to(m)

# # # Define the polygon coordinates
# polygon_coordinates = [
#     (-37.82985886920226, 145.05526523266056),
#     (-37.82968592529212, 145.05507037701463),
#     (-37.82949052083052, 145.055352996116),
#     (-37.82965509221098, 145.0555732778134)
# ]
# polygon = Polygon(polygon_coordinates)
# # folium.Polygon(locations=polygon_coordinates, color='green', fill=True, fill_color='green', fill_opacity=0.4).add_to(m)
# # # Create a Shapely polygon object
# from collections import Counter
# maid_counts = Counter()
# for lat, lon in lat_lon_list:
# # # Count how many points are inside the polygon
#     # points_inside_polygon = sum(1 for point in lat_lon_list if Point(point).within(polygon))
#     if Point(float(lat), float(lon)).within(polygon):
#         # If inside, extract the maid from the corresponding row in df_movement
#         maid = df_movement[(df_movement['latitude'] == lat) & (df_movement['longitude'] == lon)]['maid'].iloc[0]
#         # Increment the count for the maid
#         maid_counts[maid] += 1

# st.write(maid_counts)
# # Extract unique maids and their counts
# unique_maids = list(maid_counts.keys())
# maid_counts = list(maid_counts.values())

# # Display the unique maids and their counts
# for maid, count in zip(unique_maids, maid_counts):
#     st.write(f"MAID: {maid}, Count: {count}")
# # # Display the map and the count
# # st.write(f"Number of points inside the polygon: {points_inside_polygon}")
# # folium_static(m)

# st.write('unique maid count',len(unique_maids))
