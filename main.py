from download_tools import *
from preprocessing import *
from km_estimation import *

sites_southeast = ["02177000", "02178400", "03455500", "03500000", "02143500"]
sites_texas_ok = ["08167500", "08171000", "08195000", "07311500", "07332500"]
sites_mountain = ["06043500", "06191500", "09065500", "09066000", "09085000"]
sites_pnw = ["14158500", "14306500", "12189500"]
sites_newengland = ["01052500", "01055000"]

sites = sites_southeast + sites_texas_ok + sites_mountain + sites_pnw + sites_newengland

start_date = "2005-01-01"
end_date = "2026-01-01"
dt = 15

y_col = '00065' #Depth
target_folder = 'data'


def main():
    # # Step 1: Download (do once, save raw_data)
    results = {}
    raw_data = get_all_sites(sites, target_folder, start_date, end_date)
    for site_id, site_data in raw_data.items():
        y = site_data[y_col]
        site_dictionary = create_site_dictionary(y, dt= dt, omega = 0.49,c=2, std_window=50)
        compute_KM(site_dictionary)
        results[site_id] = site_dictionary