import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set page configuration
st.set_page_config(page_title="Global Weather Dashboard", layout="wide")

@st.cache_data
def load_data():
    # Attempting to load the dataset
    df = pd.read_csv('cleaned_GlobalweatherRepository.csv')
    df['date_obj'] = pd.to_datetime(df['last_updated_epoch'], unit='s')
    df['date_only'] = df['date_obj'].dt.date
    # Calculate season
    df['season'] = ((df['date_obj'].dt.month % 12 + 3) // 3).map({
        1: 'Spring', 2: 'Summer', 3: 'Autumn', 4: 'Winter'
    })
    return df

try:
    df_raw = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}. Please ensure 'cleaned_GlobalweatherRepository.csv' is in the directory.")
    st.stop()

# --- SIDEBAR FILTERS ---
st.sidebar.header("Dashboard Filters")

# Country Multi-select
all_countries = sorted(df_raw['country'].unique())
selected_countries = st.sidebar.multiselect(
    "Select Countries", 
    options=all_countries, 
    default=all_countries[:2] if len(all_countries) > 1 else all_countries
)

# Date Range Picker
min_date = df_raw['date_only'].min()
max_date = df_raw['date_only'].max()
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Latitude/Longitude Sliders
lat_min, lat_max = float(df_raw['latitude'].min()), float(df_raw['latitude'].max())
lon_min, lon_max = float(df_raw['longitude'].min()), float(df_raw['longitude'].max())

selected_lat = st.sidebar.slider("Latitude Range", lat_min, lat_max, (lat_min, lat_max))
selected_lon = st.sidebar.slider("Longitude Range", lon_min, lon_max, (lon_min, lon_max))

# --- FILTERING LOGIC ---
# Handle date_range potentially being a single value during selection
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = end_date = date_range[0] if isinstance(date_range, list) else date_range

mask = (
    df_raw['country'].isin(selected_countries) &
    (df_raw['date_only'] >= start_date) &
    (df_raw['date_only'] <= end_date) &
    (df_raw['latitude'].between(selected_lat[0], selected_lat[1])) &
    (df_raw['longitude'].between(selected_lon[0], selected_lon[1]))
)
df = df_raw[mask].copy()
df_sorted = df.sort_values('date_obj')

# --- MAIN DASHBOARD UI ---
st.title("🌍 Global Weather Analysis Dashboard")

if df.empty:
    st.warning("No data found for the selected filters. Please adjust your criteria.")
else:
    # Key Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Avg Temp", f"{df['temperature_celsius'].mean():.1f}°C")
    m2.metric("Max Temp", f"{df['temperature_celsius'].max():.1f}°C")
    m3.metric("Total Precip", f"{df['precip_mm'].sum():.1f} mm")
    m4.metric("Avg Wind", f"{df['wind_kph'].mean():.1f} kph")

    # Row 1: Temperature Over Time
    st.header("Temperature Patterns")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Yearly Avg Temperature by Country")
        fig, ax = plt.subplots()
        for country in selected_countries:
            c_data = df[df['country'] == country]
            if not c_data.empty:
                yearly_temp = c_data.groupby(c_data['date_obj'].dt.year)['temperature_celsius'].mean()
                ax.plot(yearly_temp.index, yearly_temp.values, marker='o', label=country)
        ax.set_xlabel('Year')
        ax.set_ylabel('Avg Temp (°C)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    with col2:
        st.subheader("Monthly Temperature Pattern by Country")
        fig, ax = plt.subplots()
        for country in selected_countries:
            c_data = df[df['country'] == country]
            if not c_data.empty:
                monthly_temp = c_data.groupby(c_data['date_obj'].dt.month)['temperature_celsius'].mean()
                ax.plot(monthly_temp.index, monthly_temp.values, marker='o', label=country)
        ax.set_xlabel('Month')
        ax.set_ylabel('Avg Temp (°C)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    # Row 2: Distribution and Boxplot
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Temperature Distribution (All Selected)")
        fig, ax = plt.subplots()
        ax.hist(df['temperature_celsius'], bins=30, color='skyblue', edgecolor='black')
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)

    with col4:
        st.subheader("Temperature Variance (Box Plot)")
        fig, ax = plt.subplots()
        # Boxplot by country if multiple selected
        if len(selected_countries) > 1:
            data_to_plot = [df[df['country'] == c]['temperature_celsius'] for c in selected_countries]
            ax.boxplot(data_to_plot, labels=selected_countries, patch_artist=True)
            plt.xticks(rotation=45)
        else:
            ax.boxplot(df['temperature_celsius'], vert=False, patch_artist=True, boxprops=dict(facecolor="lightgreen"))
        ax.set_xlabel('Temperature (°C)')
        st.pyplot(fig)

    # Row 3: Rainfall Analysis
    st.header("Rainfall & Precipitation")
    col5, col6 = st.columns(2)

    with col5:
        st.subheader("Total Rainfall (Yearly per Country)")
        yearly_rain = df.groupby([df['date_obj'].dt.year, 'country'])['precip_mm'].sum().unstack()
        fig, ax = plt.subplots()
        yearly_rain.plot(kind='bar', ax=ax)
        ax.set_ylabel('Total Rainfall (mm)')
        ax.set_xlabel('Year')
        ax.legend(title="Country", bbox_to_anchor=(1.05, 1))
        st.pyplot(fig)

    with col6:
        st.subheader("Rainy vs Dry Days (Total %)")
        rain_counts = (df['precip_mm'] > 0).value_counts()
        fig, ax = plt.subplots()
        ax.pie(rain_counts, labels=['Dry', 'Rainy'], autopct='%1.1f%%', 
               startangle=90, colors=['#FAD7A0', '#5DADE2'])
        st.pyplot(fig)

    # Row 4: Seasonal and Trends
    col7, col8 = st.columns(2)

    with col7:
        st.subheader("Season-wise Avg Temp by Country")
        season_order = ['Spring', 'Summer', 'Autumn', 'Winter']
        seasonal_data = df.groupby(['season', 'country'])['temperature_celsius'].mean().unstack().reindex(season_order)
        fig, ax = plt.subplots()
        seasonal_data.plot(kind='bar', ax=ax)
        ax.set_ylabel('Avg Temp (°C)')
        ax.legend(title="Country", bbox_to_anchor=(1.05, 1))
        st.pyplot(fig)

    with col8:
        st.subheader("Rolling 7-Day Temp Average")
        fig, ax = plt.subplots()
        for country in selected_countries:
            c_data = df_sorted[df_sorted['country'] == country].copy()
            c_data['temp_rolling_7'] = c_data['temperature_celsius'].rolling(7).mean()
            ax.plot(c_data['date_obj'], c_data['temp_rolling_7'], label=f'{country} (7-day)')
        ax.set_xlabel('Date')
        ax.legend(bbox_to_anchor=(1.05, 1))
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # Row 5: Heatwaves and Wind
    st.header("Extreme Events & Wind")
    col9, col10 = st.columns(2)

    with col9:
        st.subheader("Heatwave Detection (Combined 95th %)")
        threshold = df['temperature_celsius'].quantile(0.95)
        fig, ax = plt.subplots()
        for country in selected_countries:
            c_data = df[df['country'] == country]
            heatwave = c_data[c_data['temperature_celsius'] > threshold]
            ax.scatter(c_data['date_obj'], c_data['temperature_celsius'], alpha=0.2, s=10)
            ax.scatter(heatwave['date_obj'], heatwave['temperature_celsius'], label=f'{country} Heatwave', s=25)
        ax.axhline(y=threshold, color='gray', linestyle='--', label='95th % Threshold')
        ax.set_ylabel('Temp (°C)')
        ax.legend(bbox_to_anchor=(1.05, 1))
        plt.xticks(rotation=45)
        st.pyplot(fig)

    with col10:
        st.subheader("Wind Speed Trend by Country")
        fig, ax = plt.subplots()
        for country in selected_countries:
            c_data = df_sorted[df_sorted['country'] == country]
            ax.plot(c_data['date_obj'], c_data['wind_kph'], label=country, alpha=0.7)
        ax.set_ylabel('Wind Speed (kph)')
        ax.legend(bbox_to_anchor=(1.05, 1))
        plt.xticks(rotation=45)
        st.pyplot(fig)