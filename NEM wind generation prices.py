"""
NEM Wind Price Analysis Script (Fixed and Enhanced)

- Pure NEMOSIS static table only (no CSV or manual input)
- If no asset mapping table is found, prints *the contents of all static tables* so the user can debug what they have (or are missing)
- Includes a printout and instructions to fix NEMOSIS if assets table is not available
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
all_end = f"{max(years_to_run)+1}/01/01 00:00:00"
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

# --- Step 2: Wind Asset DUID/Station Assignment from static tables only ---
print("Locating VIC wind assets via package static tables ...")
from nemosis.data_fetch_methods import _defaults
static_list = _defaults.static_tables
found_plant_table = False
asset_table = None
for tab in static_list:
    try:
        test = static_table(tab, cache)
        cols = test.columns.str.upper().tolist()
        if ('DUID' in cols) and (('REGIONID' in cols) or ('REGION' in cols)) and (('UNIT_TYPE' in cols) or ('FUEL_SOURCE' in cols) or ('FUELTYPE' in cols)):
            asset_table = test
            found_plant_table = True
            print(f"Using static table: {tab}")
            break
    except Exception as e:
        print(f"Failed loading {tab}: {e}")
        continue
if not found_plant_table:
    print(f"\nNo compatible static table found in NEMOSIS install: {static_list}")
    print("\n**** DEBUG: Print ALL static tables and their columns to help you fix your install! ****")
    for tab in static_list:
        try:
            test = static_table(tab, cache)
            print(f"Static table: {tab}")
            print("Columns:", test.columns.tolist())
        except Exception as e:
            print(f"Static table: {tab} could not be loaded: {e}")
    raise Exception(f"\nCheck your nemosis/data/static folder and docs at https://github.com/UNSW-CEEM/NEMOSIS for the latest list of static tables.\nYou must ensure a static table is available (such as GENUNITS, PLANT_DATA, DUID_DESCRIPTION, or NEM_UNIT) via the package.\nIf missing, reinstall/upgrade NEMOSIS or add the required table manually.")

asset_table.columns = asset_table.columns.str.upper()
if 'FUEL_SOURCE' in asset_table.columns:
    asset_table['UNIT_TYPE'] = asset_table['FUEL_SOURCE']
if 'FUELTYPE' in asset_table.columns:
    asset_table['UNIT_TYPE'] = asset_table['FUELTYPE']

# --- VIC Wind asset DUIDs ---
wind_duids = asset_table[(asset_table['REGIONID'] == vic_region) & (asset_table['UNIT_TYPE'].str.upper().str.contains('WIND'))]['DUID'].unique().tolist()
print(f"Found {len(wind_duids)} VIC wind DUIDs from package static tables.")

# --- Auto region assignment ---
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
if 'STATION_NAME' in asset_table.columns:
    duids_df = asset_table[asset_table['DUID'].isin(wind_duids)][['DUID','STATION_NAME']].copy()
    duids_df['WindRegion'] = duids_df['STATION_NAME'].apply(region_guess)
else:
    duids_df = asset_table[asset_table['DUID'].isin(wind_duids)][['DUID']].copy()
    duids_df['WindRegion'] = 'Other'

# --- Step 3: Download Wind Generation & Calculate Price Distributions ---
all_stats = []
all_series = {}
for year in years_to_run:
    print(f'Processing {year} ...')
    prices_yr = prices[prices.index.year == year].copy()
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
    except Exception as e:
        print(f'Wind SCADA error in {year}: {e}')
        series = prices_yr.copy()

    all_series[year] = series

    plt.figure(figsize=(10,6))
    sns.kdeplot(series['RRP'], fill=True, label='All Dispatch Prices', color='grey', alpha=0.5)
    plt.title(f'VIC Price Densities ({year})')
    plt.xlabel('Spot Price ($/MWh)')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'vic_price_density_{year}.png')
    plt.close()

print("\nData processing complete. Results and plots saved.")
