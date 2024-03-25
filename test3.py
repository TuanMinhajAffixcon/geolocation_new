import folium
from folium.plugins import MarkerCluster
from folium import plugins
from streamlit_folium import folium_static
import numpy as np

# Define the center coordinates

center = (-37.82966624608861, 145.0552867255461)
# Generate random points around the center point
num_points = 500
radius = 0.001  # Assuming 1 degree of latitude/longitude is approximately 111 km
angles = np.random.uniform(0, 2*np.pi, num_points)
distances = np.random.uniform(0, radius, num_points)  # Assuming 0.001 degree is approximately 100 meters

random_points = []
for angle, distance in zip(angles, distances):
    lat = center[0] + np.sin(angle) * distance
    lon = center[1] + np.cos(angle) * distance
    random_points.append((lat, lon))
# Create a Folium Map centered at the specified location
m = folium.Map(location=center, zoom_start=15)
for point in random_points:
    folium.Circle(location=point, radius=0.01, color='blue', fill=True, fill_color='blue', fill_opacity=0.6).add_to(m)
folium.CircleMarker(location=[center[0], center[1]], radius=4, color='red', fill=True, fill_color='red',
                    fill_opacity=1).add_to(m)
# Add a circle with the specified radius around the center point
folium.Circle(
    location=center,
    radius=15,
    color='green',
    fill=True,
    fill_opacity=0.4,
).add_to(m)

# Display the map
folium_static(m)
