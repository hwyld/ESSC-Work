"""
NEM Wind Price Analysis Script (Auto-adjusted for 'Generators and Scheduled Loads')

- Uses only the static table 'Generators and Scheduled Loads' from NEMOSIS
- No CSV required, fully package-based
- Adapts column naming to match your install as printed in your debug log
- Provides wind asset matching, regional grouping, all plots/stats for wind price NPV/probabilistic analysis
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nemosis import dynamic_data_compiler, static_table
from scipy.stats import norm, kurtosis, skew
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- Config ---
cache = './nemo_cache'
vic_region = 'VIC1'
years_to_run = [2022, 2023, 2024]

if not os.path.exists(cache):
    os.makedirs(cache)

# --- Step 1: Download Spot Prices ---
print("Fetching VIC spot prices via NEMOSIS ...")
all_start = f"{min(years_to_run)}/01/01 00:00:00"
all_end   = f"{max(years_to_run)+1}/01/01 00:00:00"
prices = dynamic_data_compiler(
    all_start, all_end, 'DISPATCHPRICE', cache,
    select_columns=['SETTLEMENTDATE','REGIONID','RRP']
)
prices = prices[prices['REGIONID'] == vic_region]
prices['SETTLEMENTDATE'] = pd.to_datetime(prices['SETTLEMENTDATE'])
prices = prices.set_index('SETTLEMENTDATE').sort_index()
prices['hour'] = prices.index.hour
prices['year'] = prices.index.year
prices.to_excel('vic_spot_prices.xlsx')

# --- Step 2: Use 'Generators and Scheduled Loads' table directly ---
print("Locating VIC wind assets using 'Generators and Scheduled Loads' ...")
asset_table = static_table('Generators and Scheduled Loads', cache)

# No need to uppercase columns here; use exact field names printed by debug output
df = asset_table
# Identify wind units by either tech or fuel columns
wind_mask = (
    df['Region'] == vic_region
) & (
    df['Technology Type - Primary'].str.upper().str.contains('WIND', na=False) |
    df['Fuel Source - Primary'].str.upper().str.contains('WIND', na=False)
)
wind_duids = df[wind_mask]['DUID'].unique().tolist()
print(f"Found {len(wind_duids)} VIC wind DUIDs using 'Generators and Scheduled Loads'.")

# --- Assign region by Station Name (WesternVic, Gippsland) ---
western_keywords = ['WOOLST','MORTLAKE','PORTLAND','ARARAT','SALTCRK','MURRAH','WARRADG','YAMB']
gipps_keywords = ['HAZEL','GIPPS','GELLION','GLEN']
def region_guess(station):
    st = str(station).upper()
    if any(w in st for w in western_keywords):
        return 'WesternVic'
    elif any(g in st for g in gipps_keywords):
        return 'Gippsland'
    else:
        return 'Other'
duids_df = df[df['DUID'].isin(wind_duids)][['DUID','Station Name']].copy()
duids_df['WindRegion'] = duids_df['Station Name'].apply(region_guess)

# --- Step 3: Download Wind Generation & Calculate Price Distributions ---
all_stats = []
all_series = {}
for year in years_to_run:
    print(f'Processing {year} ...')
    prices_yr = prices[str(year)].copy()
    try:
        scada = dynamic_data_compiler(
            f'{year}/01/01 00:00:00', f'{year}/12/31 23:59:59', 'DISPATCH_UNIT_SCADA', cache,
            select_columns=['SETTLEMENTDATE', 'DUID', 'SCADAVALUE']
        )
        scada = scada[scada['DUID'].isin(wind_duids)]
        scada['SETTLEMENTDATE'] = pd.to_datetime(scada['SETTLEMENTDATE'])
        scada = scada.merge(duids_df, on='DUID', how='left')
        agg_scada = scada.groupby(['SETTLEMENTDATE', 'WindRegion'])['SCADAVALUE'].sum().unstack().fillna(0)
        agg_scada = agg_scada.resample('30T').mean()
        series = prices_yr.join(agg_scada, how='left')
        series['WesternVic_Renew_Price'] = np.where(
            series.get('WesternVic', pd.Series(index=series.index)).notnull() & (series.get('WesternVic', pd.Series(index=series.index)) > 0),
            series['RRP'],
            np.nan
        )
        if 'Gippsland' in series.columns and not series['Gippsland'].isnull().all():
            series['Gippsland_Renew_Price'] = np.where(
                series['Gippsland'].notnull() & (series['Gippsland'] > 0),
                series['RRP'],
                np.nan
            )
    except Exception as e:
        print(f'Wind SCADA error in {year}: {e}')
        series = prices_yr.copy()
    all_series[year] = series
    def price_stats(s):
        return pd.Series({
            'mean': s.mean(),
            'median': s.median(),
            'std': s.std(),
            'min': s.min(),
            'max': s.max(),
            '10%': s.quantile(0.10),
            '90%': s.quantile(0.90),
            'skew': skew(s.dropna()),
            'kurtosis': kurtosis(s.dropna())
        })
    allrow = {'Year': year, 'Type': 'All', **price_stats(series['RRP'])}
    all_stats.append(allrow)
    if 'WesternVic_Renew_Price' in series:
        wvicrow = {'Year': year, 'Type': 'WesternVic', **price_stats(series['WesternVic_Renew_Price'].dropna())}
        all_stats.append(wvicrow)
    if 'Gippsland_Renew_Price' in series:
        griprow = {'Year': year, 'Type': 'Gippsland', **price_stats(series['Gippsland_Renew_Price'].dropna())}
        all_stats.append(griprow)
    plt.figure(figsize=(10,6))
    sns.kdeplot(series['RRP'], fill=True, label='All Dispatch Prices', color='grey', alpha=0.5)
    if 'WesternVic_Renew_Price' in series:
        sns.kdeplot(series['WesternVic_Renew_Price'].dropna(), fill=True, label='WesternVic Wind-Weighted', color='blue', alpha=0.5)
    if 'Gippsland_Renew_Price' in series:
        sns.kdeplot(series['Gippsland_Renew_Price'].dropna(), fill=True, label='Gippsland Wind-Weighted', color='orange', alpha=0.5)
    plt.title(f'VIC Price Densities ({year})')
    plt.xlabel('Spot Price ($/MWh)')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'vic_price_density_{year}.png')
    plt.close()

all_stats_df = pd.DataFrame(all_stats)
all_stats_df.to_excel('vic_wind_weighted_price_stats_annual.xlsx', index=False)

plt.figure(figsize=(10,6))
for year in years_to_run:
    sns.kdeplot(all_series[year]['RRP'], label=f'All {year}', lw=2, alpha=0.8)
    if 'WesternVic_Renew_Price' in all_series[year]:
        sns.kdeplot(all_series[year]['WesternVic_Renew_Price'].dropna(), label=f'WesternVic Wind {year}', lw=2, alpha=0.8, linestyle='--')
plt.title('All Years: VIC Spot Price & Wind-Weighted Densities')
plt.xlabel('Spot Price ($/MWh)')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.savefig('vic_price_density_overlay_all_years.png')
plt.show()

print("\nAll summary stats exported to vic_wind_weighted_price_stats_annual.xlsx and density plots for all years saved.\nReady for probabilistic NPV/Monte Carlo LCOE/NPV wind farm analysis.")
