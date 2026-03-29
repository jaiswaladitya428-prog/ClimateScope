import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Page configuration
st.set_page_config(page_title="Global Weather Dashboard", layout="wide")

# Data loading function
@st.cache_data
def load_data():
    df = pd.read_csv('cleaned_GlobalweatherRepository.csv')
    df['date_obj'] = pd.to_datetime(df['last_updated_epoch'], unit='s')
    df['date_only'] = df['date_obj'].dt.date
    df['season'] = df['date_obj'].dt.month.apply(
        lambda m: 'Winter' if m in [12, 1, 2] else 'Spring' if m in [3, 4, 5] else 'Summer' if m in [6, 7, 8] else 'Autumn'
    )
    return df

# Load data
try:
    df_raw = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}. Please ensure 'cleaned_GlobalweatherRepository.csv' is in the directory.")
    st.stop()

# Create tabs
tab1, tab2 = st.tabs(["Main Dashboard", "Weather Business Insights"])

# ---------------------- TAB 1: MAIN DASHBOARD ----------------------
with tab1:
    st.title("Global Weather Analysis Dashboard")

    # Sidebar filters (only for this tab)
    st.sidebar.header("Dashboard Filters")

    # Country selection
    all_countries = sorted(df_raw['country'].unique())
    selected_countries = st.sidebar.multiselect(
        "Select Countries",
        options=all_countries,
        default=all_countries[:2] if len(all_countries) > 1 else all_countries
    )

    # Date range
    min_date = df_raw['date_only'].min()
    max_date = df_raw['date_only'].max()
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    # Latitude/Longitude
    lat_min, lat_max = float(df_raw['latitude'].min()), float(df_raw['latitude'].max())
    lon_min, lon_max = float(df_raw['longitude'].min()), float(df_raw['longitude'].max())
    selected_lat = st.sidebar.slider("Latitude Range", lat_min, lat_max, (lat_min, lat_max))
    selected_lon = st.sidebar.slider("Longitude Range", lon_min, lon_max, (lon_min, lon_max))

    # Filtering logic
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

    # Main content
    if df.empty:
        st.warning("No data found for the selected filters. Please adjust your criteria.")
    else:
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Avg Temp", f"{df['temperature_celsius'].mean():.1f}°C")
        col2.metric("Max Temp", f"{df['temperature_celsius'].max():.1f}°C")
        col3.metric("Total Precip", f"{df['precip_mm'].sum():.1f} mm")
        col4.metric("Avg Wind", f"{df['wind_kph'].mean():.1f} kph")

        # Temperature patterns
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

        # Distribution and variance
        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Temperature Distribution")
            fig, ax = plt.subplots()
            ax.hist(df['temperature_celsius'], bins=30, color='skyblue', edgecolor='black')
            ax.set_xlabel('Temperature (°C)')
            ax.set_ylabel('Frequency')
            st.pyplot(fig)

        with col4:
            st.subheader("Temperature Variance (Box Plot)")
            fig, ax = plt.subplots()
            if len(selected_countries) > 1:
                data_to_plot = [df[df['country'] == c]['temperature_celsius'] for c in selected_countries]
                ax.boxplot(data_to_plot, labels=selected_countries, patch_artist=True)
                plt.xticks(rotation=45)
            else:
                ax.boxplot(df['temperature_celsius'], vert=False, patch_artist=True, boxprops=dict(facecolor="lightgreen"))
            ax.set_xlabel('Temperature (°C)')
            st.pyplot(fig)

        # Rainfall analysis
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
            st.subheader("Rainy vs Dry Days")
            rain_counts = (df['precip_mm'] > 0).value_counts()
            fig, ax = plt.subplots()
            ax.pie(rain_counts, labels=['Dry', 'Rainy'], autopct='%1.1f%%', startangle=90, colors=['#FAD7A0', '#5DADE2'])
            st.pyplot(fig)

        # Air Quality Analysis
        if 'air_quality_pm2_5' in df.columns:
            st.header("Air Quality Analysis")
            col_aqi1, col_aqi2 = st.columns(2)

            with col_aqi1:
                st.subheader("AQI Distribution")
                fig, ax = plt.subplots()
                ax.hist(df['air_quality_pm2_5'], bins=30, color='red', edgecolor='black', alpha=0.7)
                ax.set_xlabel('PM2.5')
                ax.set_ylabel('Frequency')
                st.pyplot(fig)

            with col_aqi2:
                st.subheader("AQI by Season")
                season_order = ['Spring', 'Summer', 'Autumn', 'Winter']
                seasonal_aqi = df.groupby('season')['air_quality_pm2_5'].mean().reindex(season_order)
                fig, ax = plt.subplots()
                seasonal_aqi.plot(kind='bar', ax=ax, color='orange')
                ax.set_ylabel('Avg PM2.5')
                st.pyplot(fig)

            col_aqi3, col_aqi4 = st.columns(2)

            with col_aqi3:
                st.subheader("AQI vs Temperature Correlation")
                fig, ax = plt.subplots()
                ax.scatter(df['temperature_celsius'], df['air_quality_pm2_5'], alpha=0.6, color='purple')
                ax.set_xlabel('Temperature (°C)')
                ax.set_ylabel('PM2.5')
                st.pyplot(fig)

            with col_aqi4:
                st.subheader("AQI Trend Over Time")
                fig, ax = plt.subplots()
                for country in selected_countries:
                    c_data = df[df['country'] == country].sort_values('date_obj')
                    ax.plot(c_data['date_obj'], c_data['air_quality_pm2_5'], label=country, alpha=0.7)
                ax.set_xlabel('Date')
                ax.set_ylabel('PM2.5')
                ax.legend(bbox_to_anchor=(1.05, 1))
                plt.xticks(rotation=45)
                st.pyplot(fig)
        else:
            st.info("Air quality data (PM2.5) not available in the dataset.")

        # Seasonal and trends
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

        # Extreme events
        st.header("Extreme Events & Wind")
        col9, col10 = st.columns(2)

        with col9:
            st.subheader("Heatwave Detection")
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

# ---------------------- TAB 2: WEATHER BUSINESS INSIGHTS ----------------------
with tab2:
    st.title("Weather-Based Business Insights")

    # Checkbox to use Tab 1 data
    use_tab1_data = st.checkbox("Use filtered data from Tab 1", value=False)

    # Data source
    data_source = df if use_tab1_data and 'df' in locals() else df_raw

    if data_source.empty:
        st.warning("No data available.")
    else:
        df2 = data_source.copy()
        df2['year'] = df2['date_obj'].dt.year

        # Demand calculations
        df2['beverage_demand'] = df2['temperature_celsius'] * 2 + (df2['humidity'] * 0.5 if 'humidity' in df2.columns else 0)
        df2['ac_demand'] = df2['temperature_celsius'] * 3 + (df2['uv_index'] * 2 if 'uv_index' in df2.columns else 0)
        df2['jacket_demand'] = (40 - df2['temperature_celsius']) * 2

        # Mask demand with thresholds
        if 'air_quality_pm2_5' in df2.columns:
            df2['mask_demand'] = 1.0
            df2['mask_demand'] += (df2['air_quality_pm2_5'] > 50).astype(int) * 0.5
            df2['mask_demand'] += (df2['air_quality_pm2_5'] > 100).astype(int) * 1.0
            df2['mask_demand'] += (df2['air_quality_pm2_5'] > 150).astype(int) * 1.5
            df2['mask_demand'] += df2['date_obj'].dt.month.apply(lambda m: 0.5 if m in [11,12,1,2] else 0.2 if m in [3,4,5] else 0.1)
        else:
            df2['mask_demand'] = 1.0

        # Filters for Tab 2
        years = sorted(df2['year'].unique())
        selected_year = st.selectbox('Select Year', options=years, index=len(years)-1)

        seasons = ['Spring', 'Summer', 'Autumn', 'Winter']
        selected_season = st.selectbox('Select Season', options=seasons, index=0)

        year_df = df2[df2['year'] == selected_year]
        season_df = year_df[year_df['season'] == selected_season] if 'season' in year_df.columns else year_df

        if year_df.empty:
            st.warning(f'No records for year {selected_year}.')
        else:
            # Summary metrics
            st.subheader(f'Demand Summary for {selected_year} - {selected_season}')
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric('Avg Temp', f'{season_df["temperature_celsius"].mean():.1f} °C')
            col2.metric('Avg PM2.5', f'{season_df["air_quality_pm2_5"].mean():.1f}' if 'air_quality_pm2_5' in season_df.columns else 'N/A')
            col3.metric('Avg Mask Demand', f'{season_df["mask_demand"].mean():.2f}')
            col4.metric('Avg AC Demand', f'{season_df["ac_demand"].mean():.1f}')
            col5.metric('Avg Beverage Demand', f'{season_df["beverage_demand"].mean():.1f}')

            # Monthly trends
            st.subheader('Monthly Demand Trends (Selected Year)')
            monthly = year_df.groupby(year_df['date_obj'].dt.month).mean(numeric_only=True)
            fig, ax = plt.subplots()
            if 'beverage_demand' in monthly.columns:
                ax.plot(monthly.index, monthly['beverage_demand'], marker='o', label='Beverage')
            if 'ac_demand' in monthly.columns:
                ax.plot(monthly.index, monthly['ac_demand'], marker='o', label='AC')
            if 'jacket_demand' in monthly.columns:
                ax.plot(monthly.index, monthly['jacket_demand'], marker='o', label='Jacket')
            if 'mask_demand' in monthly.columns:
                ax.plot(monthly.index, monthly['mask_demand'], marker='o', label='Masks')
            ax.set_xlabel('Month')
            ax.set_ylabel('Index')
            ax.set_xticks(range(1, 13))
            ax.legend()
            st.pyplot(fig)

            # Sales-like bar chart
            st.subheader('Sales Proxy (Demand Indices)')
            fig2, ax2 = plt.subplots()
            ax2.bar(monthly.index - 0.2, monthly.get('beverage_demand', 0), width=0.2, label='Beverage')
            ax2.bar(monthly.index, monthly.get('ac_demand', 0), width=0.2, label='AC')
            ax2.bar(monthly.index + 0.2, monthly.get('mask_demand', 0), width=0.2, label='Masks')
            ax2.set_xlabel('Month')
            ax2.set_ylabel('Demand Index')
            ax2.set_xticks(range(1, 13))
            ax2.legend()
            st.pyplot(fig2)

            # Air quality impact
            if 'air_quality_pm2_5' in season_df.columns:
                st.subheader('Mask Demand vs Air Quality (Selected Season)')
                fig3, ax3 = plt.subplots()
                ax3.scatter(season_df['air_quality_pm2_5'], season_df['mask_demand'], alpha=0.6)
                ax3.set_xlabel('PM2.5')
                ax3.set_ylabel('Mask Demand Index')
                st.pyplot(fig3)

            # Seasonal air quality
            st.subheader('Seasonal Air Quality Impact')
            if 'season' in df2.columns:
                season_quality = year_df.groupby('season')['air_quality_pm2_5'].mean().reindex(seasons)
                fig4, ax4 = plt.subplots()
                season_quality.plot(kind='bar', ax=ax4, color='teal')
                ax4.set_ylabel('Avg PM2.5')
                st.pyplot(fig4)

            # Top countries
            st.subheader('Top Countries by Mask Demand (Selected Year)')
            if 'country' in year_df.columns and 'mask_demand' in year_df.columns:
                top_countries = year_df.groupby('country')['mask_demand'].mean().sort_values(ascending=False).head(10)
                st.table(top_countries.reset_index().rename(columns={'mask_demand': 'Avg Mask Demand'}))

            # Insights
            st.subheader('Insights')
            st.write('''
            - Mask demand increases with PM2.5 levels and is higher in winter.
            - Beverage and AC demand correlate with temperature.
            - Use the checkbox to apply Tab 1 filters here if desired.
            ''')