import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the data
try:
    solar_df = pd.read_csv('CF Solar.csv')
    wind_on_df = pd.read_csv('CF Wind Onshore.csv')
    wind_off_df = pd.read_csv('CF Wind Offshore.csv')
except FileNotFoundError as e:
    print(f"Error: {e}. Please check your file names and paths.")
    exit()

# 2. Preprocess function to clean Datetime and extract 'DK'
def preprocess_df(df, name):
    df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True)
    df = df[['Datetime', 'DK']].rename(columns={'DK': name})
    return df

solar = preprocess_df(solar_df, 'Solar')
wind_on = preprocess_df(wind_on_df, 'Wind Onshore')
wind_off = preprocess_df(wind_off_df, 'Wind Offshore')

# Merge and create Total Wind
data = solar.merge(wind_on, on='Datetime').merge(wind_off, on='Datetime')
data['Total Wind'] = (data['Wind Onshore'] + data['Wind Offshore']) / 2
data.set_index('Datetime', inplace=True)

# -----------------------------------------------------------------
# VISUALIZATION - SIDE-BY-SIDE LAYOUT
# -----------------------------------------------------------------
# 1 row, 2 columns layout
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Unified style colors
color_wind = '#1F77B4'   # Blue
color_solar = '#FFA500'  # Orange

# =================================================================
# LEFT PLOT: Seasonal / Annual Complementarity (Monthly Averages)
# =================================================================
seasonal_data = data.groupby(data.index.month).mean()
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Plot lines
axes[0].plot(months, seasonal_data['Total Wind'], label='Wind', color=color_wind, linewidth=1.5)
axes[0].plot(months, seasonal_data['Solar'], label='Solar PV', color=color_solar, linewidth=1.5)

# Fill areas underneath
axes[0].fill_between(months, seasonal_data['Total Wind'], color=color_wind, alpha=0.15)
axes[0].fill_between(months, seasonal_data['Solar'], color=color_solar, alpha=0.25)

# Format left plot
axes[0].set_title('Seasonal Capacity Factors in Denmark (2018)', fontsize=13, pad=15)
axes[0].set_ylabel('Capacity Factor', fontsize=11)
axes[0].set_xlabel('Month', fontsize=11)
axes[0].grid(True, linestyle='--', alpha=0.3, color='#E0E0E0')
axes[0].legend(loc='upper right', fontsize=9)

# Clean up borders (spines)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].spines['left'].set_color('#666666')
axes[0].spines['bottom'].set_color('#666666')


# =================================================================
# RIGHT PLOT: Daily Profile (Hourly Averages ONLY for May)
# =================================================================
target_month = 5  # 5 represents May
filtered_data = data[data.index.month == target_month]

# Group using the FILTERED data's index to avoid ValueError
daily_profile = filtered_data.groupby(filtered_data.index.hour).mean()
hours = daily_profile.index

# Plot lines
axes[1].plot(hours, daily_profile['Total Wind'], label='Wind', color=color_wind, linewidth=1.5)
axes[1].plot(hours, daily_profile['Solar'], label='Solar PV', color=color_solar, linewidth=1.5)

# Fill areas underneath
axes[1].fill_between(hours, daily_profile['Total Wind'], color=color_wind, alpha=0.15)
axes[1].fill_between(hours, daily_profile['Solar'], color=color_solar, alpha=0.25)

# Format right plot
axes[1].set_title('Daily Capacity Factors Profile in Denmark in May (2018)', fontsize=13, pad=15)
axes[1].set_ylabel('Capacity Factor', fontsize=11)
axes[1].set_xlabel('Hour of the Day', fontsize=11)
axes[1].set_xticks(range(0, 24, 3))  # Show labels every 3 hours for neat spacing
axes[1].grid(True, linestyle='--', alpha=0.3, color='#E0E0E0')
axes[1].legend(loc='upper right', fontsize=9)

# Clean up borders (spines)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].spines['left'].set_color('#666666')
axes[1].spines['bottom'].set_color('#666666')


# Auto-adjust subplot padding and render
plt.tight_layout()
plt.savefig('pv_wind_complementarity.png', dpi=300)  # Save the figure as a high-res PNG
plt.show()