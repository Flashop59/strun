import pandas as pd
import numpy as np
from shapely.geometry import Polygon
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
import folium
from folium import plugins
from geopy.distance import geodesic
from datetime import datetime, timedelta
from IPython.display import display
from IPython.core.display import HTML
import requests
import streamlit as st

# Function to fetch data from the API
def fetch_data(vehicle, start_time, end_time):
    API_KEY = "3330d953-7abc-4bac-b862-ac315c8e2387-6252fa58-d2c2-4c13-b23e-59cefafa4d7d"
    url = f"https://admintestapi.ensuresystem.in/api/locationpull/orbit?vehicle={vehicle}&from={start_time}&to={end_time}"
    headers = {"token": API_KEY}
    
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Error fetching data: {response.status_code}")
        return None

    data = response.json()
    if not isinstance(data, list):
        print(f"Unexpected data format: {data}")
        return None
    
    # Sort data by time
    data.sort(key=lambda x: x['time'])
    return data

# Function to calculate the area of a field in square meters using convex hull
def calculate_convex_hull_area(points):
    if len(points) < 3:  # Not enough points to form a polygon
        return 0
    try:
        hull = ConvexHull(points)
        poly = Polygon(points[hull.vertices])
        return poly.area  # Area in square degrees
    except Exception:
        return 0

# Function to calculate centroid of a set of points
def calculate_centroid(points):
    return np.mean(points, axis=0)

# Function to generate more points along the convex hull
def generate_more_hull_points(points, num_splits=3):
    new_points = []
    for i in range(len(points)):
        start_point = points[i]
        end_point = points[(i + 1) % len(points)]
        new_points.append(start_point)
        for j in range(1, num_splits):
            intermediate_point = start_point + j * (end_point - start_point) / num_splits
            new_points.append(intermediate_point)
    return np.array(new_points)

# Function to process the fetched data and return the map and field areas
def process_data(data):
    # Create a DataFrame from the fetched data
    gps_data = pd.DataFrame(data)
    gps_data['Timestamp'] = pd.to_datetime(gps_data['time'], unit='ms')
    gps_data['lat'] = gps_data['lat']
    gps_data['lng'] = gps_data['lon']
    
    # Cluster the GPS points to identify separate fields
    coords = gps_data[['lat', 'lng']].values
    db = DBSCAN(eps=0.00008, min_samples=11).fit(coords)
    labels = db.labels_

    # Add labels to the data
    gps_data['field_id'] = labels

    # Calculate the area for each field
    fields = gps_data[gps_data['field_id'] != -1]  # Exclude noise points
    field_areas = fields.groupby('field_id').apply(
        lambda df: calculate_convex_hull_area(df[['lat', 'lng']].values))

    # Convert the area from square degrees to square meters (approximation)
    field_areas_m2 = field_areas * 0.77 * (111000 ** 2)  # rough approximation

    # Convert the area from square meters to gunthas (1 guntha = 101.17 m^2)
    field_areas_gunthas = field_areas_m2 / 101.17

    # Calculate time metrics for each field
    field_times = fields.groupby('field_id').apply(
        lambda df: (df['Timestamp'].max() - df['Timestamp'].min()).total_seconds() / 60.0
    )

    # Extract start and end dates for each field
    field_dates = fields.groupby('field_id').agg(
        start_date=('Timestamp', 'min'),
        end_date=('Timestamp', 'max')
    )

    # Filter out fields with area less than 5 gunthas
    valid_fields = field_areas_gunthas[field_areas_gunthas >= 5].index
    field_areas_gunthas = field_areas_gunthas[valid_fields]
    field_times = field_times[valid_fields]
    field_dates = field_dates.loc[valid_fields]

    # Calculate centroids of each field
    centroids = fields.groupby('field_id').apply(
        lambda df: calculate_centroid(df[['lat', 'lng']].values)
    )

    # Calculate traveling distance and time between field centroids
    travel_distances = []
    travel_times = []
    field_ids = list(valid_fields)
    
    if len(field_ids) > 1:
        for i in range(len(field_ids) - 1):
            centroid1 = centroids.loc[field_ids[i]]
            centroid2 = centroids.loc[field_ids[i + 1]]
            distance = geodesic(centroid1, centroid2).kilometers
            time = (field_dates.loc[field_ids[i + 1], 'start_date'] - field_dates.loc[field_ids[i], 'end_date']).total_seconds() / 60.0
            travel_distances.append(distance)
            travel_times.append(time)

        # Calculate distance from last point of one field to first point of the next field
        for i in range(len(field_ids) - 1):
            end_point = fields[fields['field_id'] == field_ids[i]][['lat', 'lng']].values[-1]
            start_point = fields[fields['field_id'] == field_ids[i + 1]][['lat', 'lng']].values[0]
            distance = geodesic(end_point, start_point).kilometers
            time = (field_dates.loc[field_ids[i + 1], 'start_date'] - field_dates.loc[field_ids[i], 'end_date']).total_seconds() / 60.0
            travel_distances.append(distance)
            travel_times.append(time)

        # Append NaN for the last field
        travel_distances.append(np.nan)
        travel_times.append(np.nan)
    else:
        travel_distances.append(np.nan)
        travel_times.append(np.nan)

    # Ensure lengths match for DataFrame
    if len(travel_distances) != len(field_areas_gunthas):
        travel_distances = travel_distances[:len(field_areas_gunthas)]
        travel_times = travel_times[:len(field_areas_gunthas)]

    # Combine area, time, dates, and travel metrics into a single DataFrame
    combined_df = pd.DataFrame({
        'Field ID': field_areas_gunthas.index,
        'Area (Gunthas)': field_areas_gunthas.values,
        'Time (Minutes)': field_times.values,
        'Start Date': field_dates['start_date'].values,
        'End Date': field_dates['end_date'].values,
        'Travel Distance to Next Field (km)': travel_distances,
        'Travel Time to Next Field (minutes)': travel_times
    })
    
    # Calculate total metrics
    total_area = field_areas_gunthas.sum()
    total_time = field_times.sum()
    total_travel_distance = np.nansum(travel_distances)
    total_travel_time = np.nansum(travel_times)

    # Create a satellite map
    map_center = [gps_data['lat'].mean(), gps_data['lng'].mean()]
    m = folium.Map(location=map_center, zoom_start=12)
    
    # Add Mapbox satellite imagery
    mapbox_token = 'pk.eyJ1IjoiZmxhc2hvcDAwNyIsImEiOiJjbHo5NzkycmIwN2RxMmtzZHZvNWpjYmQ2In0.A_FZYl5zKjwSZpJuP_MHiA'
    folium.TileLayer(
        tiles='https://api.mapbox.com/styles/v1/mapbox/satellite-v9/tiles/256/{z}/{x}/{y}?access_token=' + mapbox_token,
        attr='Mapbox Satellite Imagery',
        name='Satellite',
        overlay=True,
        control=True
    ).add_to(m)
    
    # Add fullscreen control
    plugins.Fullscreen(position='topright').add_to(m)

    # Plot the points on the map
    for idx, row in gps_data.iterrows():
        if row['field_id'] in valid_fields:
            color = 'blue'  # Blue for valid field points
        else:
            color = 'red'   # Red for travel points (noise)
        folium.CircleMarker(
            location=(row['lat'], row['lng']),
            radius=2,
            color=color,
            fill=True,
            fill_color=color
        ).add_to(m)

    # Plot convex hulls for each valid field and display hull points in yellow
    for field_id in valid_fields:
        field_points = fields[fields['field_id'] == field_id][['lat', 'lng']].values
        hull = ConvexHull(field_points)
        hull_points = field_points[hull.vertices]
        
        print(f"Field ID {field_id} Hull Points:")
        for point in hull_points:
            print(f"Lat: {point[0]}, Lng: {point[1]}")
            folium.CircleMarker(
                location=(point[0], point[1]),
                radius=4,
                color='yellow',
                fill=True,
                fill_color='yellow'
            ).add_to(m)
        
        # Generate more hull points and plot them
        extended_hull_points = generate_more_hull_points(hull_points)
        for i in range(len(extended_hull_points)):
            folium.CircleMarker(
                location=(extended_hull_points[i][0], extended_hull_points[i][1]),
                radius=3,
                color='yellow',
                fill=True,
                fill_color='yellow'
            ).add_to(m)

    return m, combined_df, total_area, total_time, total_travel_distance, total_travel_time

# Main code in Streamlit
st.title("Vehicle Field Area and Time Analysis")

vehicle = st.text_input("Enter Vehicle ID (e.g., BR1):")
start_date = st.date_input("Enter Start Date:", datetime.now())
end_date = st.date_input("Enter End Date:", datetime.now() + timedelta(days=1))

if st.button("Fetch and Analyze Data"):
    start_time = int(start_date.timestamp() * 1000)
    end_time = int(end_date.timestamp() * 1000)

    data = fetch_data(vehicle, start_time, end_time)

    if data:
        map_obj, field_df, total_area, total_time, total_travel_distance, total_travel_time = process_data(data)
        
        # Display the map
        folium_static(map_obj)

        # Display the DataFrame and totals
        st.dataframe(field_df)
        
        st.write(f"Total Area (Gunthas): {total_area}")
        st.write(f"Total Time (Minutes): {total_time}")
        st.write(f"Total Travel Distance (km): {total_travel_distance}")
        st.write(f"Total Travel Time (minutes): {total_travel_time}")
        
        # Add button to download the map as HTML
        st.download_button(
            label="Download Map as HTML",
            data=map_obj.get_root().render(),
            file_name="map.html",
            mime="text/html"
        )
