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

# # Define the polygon coordinates
# polygon_coordinates = [
#     (-37.82985886920226, 145.05526523266056),
#     (-37.82968592529212, 145.05507037701463),
#     (-37.82949052083052, 145.055352996116),
#     (-37.82965509221098, 145.0555732778134)
# ]
# polygon = Polygon(polygon_coordinates)

# unique_maids = set()  # Initialize an empty set to store unique maids

# for i in range(len(location)):
#     # Extract latitude and longitude from location dataframe
#     lat = location['user_lat'][i]
#     lon = location['user_lon'][i]
#     # Check if the point is inside the polygon
#     if Point(lon, lat).within(polygon):
#         # Query to get unique maids inside the polygon area
#         mycursor.execute(f"SELECT DISTINCT maid FROM lifesight.tbl_movement_geohash_parquet \
#                           WHERE month='{current_month}' AND year='{current_year}' \
#                           AND latitude={lat} AND longitude={lon} \
#                           AND CAST(SUBSTRING(datetimestamp,12,2) AS INTEGER) BETWEEN 13 AND 21")
#         # Fetch results
#         maids_inside_polygon = mycursor.fetchall()
#         # Add unique maids to the set
#         for maid in maids_inside_polygon:
#             unique_maids.add(maid[0])

# # Count the number of unique maids
# unique_maid_count = len(unique_maids)

# # Display the count of unique maids
# st.write(f"Number of unique maids inside the polygon: {unique_maid_count}")

import numpy as np
import streamlit as st
import pandas as pd

center_point = (-37.82967324035952, 145.05530846706537)
# Generate random points around the center point
num_points = 500
radius = 0.001  # Assuming 1 degree of latitude/longitude is approximately 111 km
angles = np.random.uniform(0, 2*np.pi, num_points)
distances = np.random.uniform(0, radius, num_points)  # Assuming 0.001 degree is approximately 100 meters

random_points = []
for angle, distance in zip(angles, distances):
    lat = center_point[0] + np.sin(angle) * distance
    lon = center_point[1] + np.cos(angle) * distance
    random_points.append((lat, lon))

st.write(pd.DataFrame(random_points))