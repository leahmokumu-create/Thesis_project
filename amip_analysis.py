# run_amip_analysis.py
import xarray as xr
import sys
from pathlib import Path
from collections import defaultdict
import config
import analysis_lib as alib

# =============================================================================
# 1. OBS DATA LOADER (Wrapper for lib function)
# =============================================================================
def get_obs_sst_data():
    """
    Wraps the library function to get the Observed SST needed for AMIP forcing.
    Returns:
        land_mask: For masking the model data
        obs_sst_ocean: The FULL RANGE (1950-2020) observed SST to use for indices
    """
    print("--- (AMIP Setup) Loading Observed HadISST for Forcing/Teleconnections ---")
    # We ignore the 3rd return value (obs precip) as we don't need it for AMIP processing
    land_mask, obs_sst_ocean, _ = alib.load_processed_obs(load_full_range=True)
    return land_mask, obs_sst_ocean

# =============================================================================
# 2. MODEL PROCESSOR
# =============================================================================
def process_amip_model(model_name, model_file_list, land_mask, obs_sst_ocean):
    """
    Process a single AMIP model using the centralized analysis loop.
    """
    print(f"\n{'='*50}\n--- Processing Model: {model_name} ---\n{'='*50}")
    
    try:
        # --- 1. Load and Standardize Model Precip ---
        time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
        with xr.open_mfdataset(model_file_list, combine='by_coords', decode_times=time_coder) as ds:
            pr_std = alib.standardize_coords(alib.align_calendar(ds))
            # Slice 1950-2020 (TS Range)
            pr_data = pr_std.sel(time=slice(str(config.TS_START_YEAR), str(config.TS_END_YEAR)))
            
            # Find precip variable name
            if 'pr' in pr_data.data_vars: 
                pr_var = 'pr'
            else: 
                pr_var = [v for v in pr_data.data_vars if v not in pr_data.dims][0]
        
        # --- 2. Regrid (Conservative) ---
        print(f"  -> Regridding {model_name} precip...")
        w_file = config.OUTPUT_DIR / 'regridder_weights' / f"weights_amip_{model_name}_pr_conservative.nc"
        
        # Use the lib function to get the regridder (handles creation/loading)
        regridder = alib.get_regridder(pr_data, "conservative", w_file)
        
        # Apply regridder and mask
        pr_land = regridder(pr_data[pr_var]).where(land_mask)
        
        # --- 3. Convert Units ---
        pr_land = alib.convert_precip_units(pr_land).persist()

        # --- 4. Run Centralized Analysis Loop ---
        # We pass 'obs_sst_ocean' here because AMIP models are forced by observed SSTs.
        # This replaces the huge blocks of repeated code (A, B, C, D).
        alib.run_core_analysis_loop(
            pr_data=pr_land, 
            sst_data_for_indices=obs_sst_ocean, 
            output_prefix=f"amip_{model_name}", 
            land_mask=land_mask
        )
        
        print(f"✅ Successfully finished {model_name}")
        return model_name

    except Exception as e:
        print(f"❌ ERROR processing {model_name}: {e}")
        # import traceback; traceback.print_exc() # Uncomment for deeper debugging
        return None

# =============================================================================
# 3. MAIN
# =============================================================================
def main():
    print("--- 1. Starting AMIP Analysis ---")
    
    # 1. Load Obs Data
    land_mask, obs_sst_ocean = get_obs_sst_data()
    
    # 2. Find Files
    print(f"\n--- 2. Searching for AMIP models in: {config.AMIP_BASE_DIR} ---")
    amip_base_path = str(config.AMIP_BASE_DIR)
    all_files = sorted(Path(amip_base_path).glob('*.nc'))
    
    if not all_files:
        print(f"❌ CRITICAL ERROR: No .nc files found in {config.AMIP_BASE_DIR}")
        sys.exit()
    
    model_groups = defaultdict(list)
    for fpath in all_files:
        try:
            # Extracts 'Access-CM2' from 'Access-CM2_amip_...'
            model_name_base = fpath.stem.rsplit('_', 1)[0] 
            model_groups[model_name_base].append(fpath)
        except Exception as e:
            print(f"Could not parse model name for {fpath.name}: {e}")
            
    print(f"  -> Found {len(all_files)} files, grouped into {len(model_groups)} models.")

    # 3. Process Models
    successful_models = []
    for model_name, file_list in model_groups.items():
        # !!! This is the fix: passing obs_sst_ocean explicitly !!!
        result = process_amip_model(model_name, file_list, land_mask, obs_sst_ocean)
        if result: 
            successful_models.append(result)

    print(f"\n--- 🏁 AMIP Analysis Complete ---")
    print(f"Successfully processed {len(successful_models)} out of {len(model_groups)} models.")

if __name__ == "__main__":
    main()