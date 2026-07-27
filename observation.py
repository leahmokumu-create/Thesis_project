# observation.py
import analysis_lib as alib
from unittest.mock import patch

def main():
    print("--- Starting Observational Analysis ---")

    
    real_convert = alib.convert_precip_units

    def safe_convert_precip_units(da):
        # 1. Check if the 'units' attribute is missing
        if 'units' not in da.attrs:
            print("    [Patch] Missing units detected in GPCC data. Temporarily assigning 'mm/month'.")
            da.attrs['units'] = 'mm/month'
        
        # 2. Pass the corrected data to the real library function
        return real_convert(da)
    # ------------------------------------------------------------------------

    print("--- (Local) Running Analysis with Safety Patch ---")

    # Apply the patch ONLY for the duration of this loading step
    with patch('analysis_lib.convert_precip_units', side_effect=safe_convert_precip_units):
        
        # 1. Load Data (Standard Library Call)
        land_mask, sst_ocean, pr_land = alib.load_processed_obs(load_full_range=True)
    
    # 2. Run Analysis (Standard Library Call)
    alib.run_core_analysis_loop(
        pr_data=pr_land, 
        sst_data_for_indices=sst_ocean, 
        output_prefix="obs", 
        land_mask=land_mask
    )

    print("--- 🏁 Observational Analysis Complete ---")

if __name__ == "__main__":
    main()