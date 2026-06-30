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

# 2. Preprocess function
def preprocess_df(df, name):
    df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True)
    df = df[['Datetime', 'DK']].rename(columns={'DK': name})
    return df

solar = preprocess_df(solar_df, 'Solar')
wind_on = preprocess_df(wind_on_df, 'Wind Onshore')
wind_off = preprocess_df(wind_off_df, 'Wind Offshore')

# Merge and create Total Wind
data = solar.merge(wind_on, on='Datetime').merge(wind_off, on='Datetime')
data['Total Wind'] = data['Wind Onshore'] + data['Wind Offshore']
data.set_index('Datetime', inplace=True)


# --- FILTER FOR A SPECIFIC MONTH AND PLOT DAILY PROFILE ---
target_month = 5 
month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
               7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}

# 1. Filter the rows down to the target month
filtered_data = data[data.index.month == target_month]

# 2. Group using the FILTERED data's index (Fixes the ValueError)
daily_profile = filtered_data.groupby(filtered_data.index.hour).mean()
hours = daily_profile.index

# Define colors (Blue for wind, Orange for solar)
color_wind = '#1F77B4'
color_solar = '#FFA500'

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot lines
ax.plot(hours, daily_profile['Total Wind'], label='Wind (Total)', color=color_wind, linewidth=2)
ax.plot(hours, daily_profile['Solar'], label='Solar PV', color=color_solar, linewidth=2)

# Fill areas underneath
ax.fill_between(hours, daily_profile['Total Wind'], color=color_wind, alpha=0.15)
ax.fill_between(hours, daily_profile['Solar'], color=color_solar, alpha=0.25)

# Formatting
ax.set_title(f'Average Daily Profile in Denmark ({month_names[target_month]})', fontsize=14, pad=15, fontweight='bold')
ax.set_ylabel('Capacity Factor', fontsize=12)
ax.set_xlabel('Hour of the Day', fontsize=12)
ax.set_xticks(range(0, 24, 2))  # Ticks every 2 hours
ax.grid(True, linestyle='--', alpha=0.3, color='#E0E0E0')
ax.legend(loc='upper right', fontsize=11)

# Clean up borders (spines)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#666666')
ax.spines['bottom'].set_color('#666666')

plt.tight_layout()
plt.show()