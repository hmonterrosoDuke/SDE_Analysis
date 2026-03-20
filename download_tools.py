

from dataclasses import dataclass
from typing import Optional, Dict, Any
import pandas as pd
import geopandas as gpd
from dataretrieval import nwis
import os

@dataclass
class Basin:
    """Container for all basin-related data."""
    site_id: str
    metadata: Dict[str, Any]
    polygon: Optional[gpd.GeoDataFrame] = None
    characteristics: Optional[pd.DataFrame] = None
    timeseries: Optional[pd.DataFrame] = None

def download_iv(site: str, start: str, end: str) -> pd.DataFrame:
    
    df = nwis.get_record(sites=site, service="iv", start=start, end=end)
    
    if df is None or df.empty:
        raise ValueError(f"No data found for site {site}")
    
    return df


def get_gage_height(df: pd.DataFrame) -> pd.Series:
    """
    Extract gage height (parameter 00065) from IV dataframe.
    """
    cols = [c for c in df.columns if "00065" in c and "_cd" not in c]
    
    if not cols:
        raise ValueError("No gage height column found")
    
    return df[cols[0]].to_numpy()


def get_all_sites(site_ids, target_folder, start_date, end_date):
    """get data for multiple sites. Download if site is not in csv saved
    
    Returns
    -------
    raw_data : dict
        {site_id: raw_dataframe}
    """
    raw_data = {}
    
    for site_id in site_ids:
        file_name = site_id +'.csv'
        path = os.path.join(target_folder, file_name)

        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            # y = get_gage_height(df)
            raw_data[site_id] = df

        else: 
            print(f"Downloading {site_id}...")
            try:
                site_data = download_iv(site_id, start_date, end_date)
                # y = get_gage_height(site_data)
                raw_data[site_id] = site_data
                print(f"  ✓ {len(site_data)} observations")

                site_data.to_csv(path)
            except Exception as e:
                print(f"  ✗ Failed: {e}")

    return raw_data