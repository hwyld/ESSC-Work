import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nemosis import dynamic_data_compiler

# --- User settings ---
start = '2022/01/01 00:00:00'
end   = '2024/01/01 00:00:00'
cache = './nemo_cache'
vic_region = 'VIC1'
wind_station = 'PORTLANDWF1'  # Or another DUID of your site

# --- Ensure cache directory exists ---
if not os.path.exists(cache):
    os.makedirs(cache)

# --- Download spot prices ---
print("Downloading NEM spot prices...")
prices = dynamic_data_compiler(
    start, end, 'DISPATCHPRICE', cache,
    select_columns=['SETTLEMENTDATE','REGIONID','RRP']
)
prices = prices[prices['REGIONID'] == vic_region]
prices['SETTLEMENTDATE'] = pd.to_datetime(prices['SETTLEMENTDATE'])
prices = prices.set_index('SETTLEMENTDATE').sort_index()

# --- Download demand and available generation ---
print("Downloading NEM demand and available generation...")
gen = dynamic_data_compiler(
    start, end, 'DISPATCHREGIONSUM', cache,
    select_columns=['SETTLEMENTDATE','REGIONID','TOTALDEMAND','AVAILABLE_GENERATION']
)
gen = gen[gen['REGIONID'] == vic_region]
gen['SETTLEMENTDATE'] = pd.to_datetime(gen['SETTLEMENTDATE'])
gen = gen.set_index('SETTLEMENTDATE').sort_index()

# --- Download wind SCADA data (optional) ---
try:
    print("Downloading wind SCADA data...")
    scada = dynamic_data_compiler(
        start, end, 'DISPATCH_UNIT_SCADA', cache,
        select_columns=['SETTLEMENTDATE','DUID','SCADAVALUE']
    )
    wind = scada[scada['DUID'] == wind_station]
    wind['SETTLEMENTDATE'] = pd.to_datetime(wind['SETTLEMENTDATE'])
    wind = wind.set_index('SETTLEMENTDATE').sort_index()
    wind = wind.rename(columns={'SCADAVALUE': 'wind_mw'})
except Exception as e:
    print(f"SCADA data for wind not found: {e}")
    wind = None

# --- Merge all datasets ---
dfs = [prices[['RRP']], gen[['TOTALDEMAND','AVAILABLE_GENERATION']]]
if wind is not None and not wind.empty:
    dfs.append(wind[['wind_mw']])

df = pd.concat(dfs, axis=1)
df = df.dropna()
df.to_excel('nem_vic_hourly_data.xlsx')
print("Merged data saved to nem_vic_hourly_data.xlsx")

# --- Add hour-of-day and year columns ---
df['hour'] = df.index.hour
df['year'] = df.index.year

# --- Plot: Histogram of annual price distribution for each hour ---
years = sorted(df['year'].unique())
for yr in years:
    plt.figure(figsize=(16, 10))
    for hr in range(24):
        plt.subplot(4, 6, hr+1)
        plt.hist(df.loc[(df['hour'] == hr) & (df['year'] == yr), 'RRP'], bins=50, alpha=0.7)
        plt.title(f"Hour {hr}")
        plt.xticks([])
        plt.yticks([])
    plt.suptitle(f'NEM VIC Spot Price Distribution by Hour, {yr}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'nem_vic_price_histogram_hourly_{yr}.png')
    plt.show()

print("Histograms saved for each year.")

# ---- HOW TO INCORPORATE INTO LCOE MODELLING ----
print("""
How to use these price distributions in LCOE modelling:

1. For each hour, you have a distribution of possible market prices (from history).
2. If you have an hourly generation profile for your wind farm (from SCADA or Renewables.ninja), 
   multiply the plant's expected hourly output by a sampled price from the corresponding hour's distribution.
3. Repeat for all hours in the year to estimate annual revenue. Run Monte Carlo simulations (sample price each hour).
4. Subtract annual fixed and variable costs to get net cashflow, then divide by total annual MWh for LCOE.
5. Repeat for many simulations to build a probability distribution for LCOE and investment returns (e.g., VaR).

Example pseudo-code:
for sim in range(N):
    revenue = 0
    for hour in range(8760):
        gen = hourly_generation_profile[hour]
        price = np.random.choice(price_histogram_by_hour[hour])
        revenue += gen * price
    lcoe = (total_costs) / (total_annual_mwh)
    results.append(lcoe)
""")
