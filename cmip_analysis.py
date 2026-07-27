# run_cmip_analysis.py
import xarray as xr
import sys
from pathlib import Path
import config
import analysis_lib as alib

# =============================================================================
# 1. CMIP MODEL DATA FINDER
# =============================================================================
def find_latest_data_path(base_path):
    try:
        grid_folder = list(base_path.iterdir())[0]
        if not grid_folder.is_dir(): return None
        version_folders = sorted([d for d in grid_folder.iterdir() if d.is_dir()])
        if not version_folders: return None
        latest_version_path = version_folders[-1]
        return str(latest_version_path / '*.nc')
    except (IndexError, FileNotFoundError):
        return None

def find_cmip_data():
    print(f"--- 1. Searching for CMIP data in: {config.CMIP_BASE_DIR} ---")
    data_paths = {} 
    for model_dir in sorted(config.CMIP_BASE_DIR.glob('*/*')):
        if model_dir.is_dir():
            model_name = model_dir.name
            pr_base_path = model_dir / 'historical' / 'r1i1p1f1' / 'Amon' / 'pr'
            tos_base_path = model_dir / 'historical' / 'r1i1p1f1' / 'Omon' / 'tos'
            
            if pr_base_path.exists() and tos_base_path.exists():
                pr_latest_path = find_latest_data_path(pr_base_path)
                tos_latest_path = find_latest_data_path(tos_base_path)
                
                if pr_latest_path and tos_latest_path:
                    data_paths[model_name] = {
                        'pr_hist': pr_latest_path, 
                        'tos_hist': tos_latest_path,
                        'tos_var_name': 'tos', 
                        'pr_var_name': 'pr'
                    }
                    print(f"  [+] Found data for model: {model_name}")
    
    if not data_paths:
        print("\nCRITICAL ERROR: No models with both 'pr' and 'tos' (r1i1p1f1) found.")
        sys.exit() 
    print(f"\n--- Found {len(data_paths)} valid models. ---")
    return data_paths

# =============================================================================
# 2. MAIN PROCESSING LOOP
# =============================================================================
def main():
    cmip_data_paths = find_cmip_data()
    print("\n--- Loading Universal Land Mask ---")
    land_mask = alib.get_universal_land_mask()
    
    successful_models = []
    
    for model_name, paths in cmip_data_paths.items():
        print(f"\n{'='*50}\n--- Processing Model: {model_name} ---\n{'='*50}")
        try:
            # --- A. Load and Standardize Data ---
            time_coder = xr.coders.CFDatetimeCoder(use_cftime=True) 
            
            # Load Precip
            with xr.open_mfdataset(paths['pr_hist'], chunks='auto', decode_times=False) as ds_pr:
                ds_pr_decoded = xr.decode_cf(ds_pr, decode_times=time_coder)
                pr_std = alib.standardize_coords(alib.align_calendar(ds_pr_decoded))
                pr_data = pr_std.sel(time=slice(str(config.TS_START_YEAR), str(config.TS_END_YEAR)))
                pr_var = paths['pr_var_name']

            # Load SST (tos)
            with xr.open_mfdataset(paths['tos_hist'], chunks='auto', decode_times=False) as ds_tos:
                ds_tos_decoded = xr.decode_cf(ds_tos, decode_times=time_coder)
                tos_std = alib.standardize_coords(alib.align_calendar(ds_tos_decoded))
                tos_data = tos_std.sel(time=slice(str(config.TS_START_YEAR), str(config.TS_END_YEAR)))
                tos_var = paths['tos_var_name']
            
            # =================================================================
            # RESTORED SANITY CHECK: SKIP UNSTRUCTURED GRIDS (e.g., AWI)
            # =================================================================
            # Check Precip Grid
            SANITY_THRESHOLD = 5000
            if ('lat' in pr_data.coords and pr_data.lat.ndim == 1 and len(pr_data.lat) > SANITY_THRESHOLD) or \
               ('lon' in pr_data.coords and pr_data.lon.ndim == 1 and len(pr_data.lon) > SANITY_THRESHOLD):
                print(f"  [!] Skipping {model_name}. 'pr' grid is a 1D unstructured mesh (too large for standard regridding).")
                continue
            
            # Check SST Grid
            if ('lat' in tos_data.coords and tos_data.lat.ndim == 1 and len(tos_data.lat) > SANITY_THRESHOLD) or \
               ('lon' in tos_data.coords and tos_data.lon.ndim == 1 and len(tos_data.lon) > SANITY_THRESHOLD):
                print(f"  [!] Skipping {model_name}. 'tos' grid is a 1D unstructured mesh (too large for standard regridding).")
                continue
            # =================================================================

            # --- B. Regrid ---
            print("  -> Regridding data...")
            # Precip -> Conservative
            pr_w_file = config.OUTPUT_DIR / 'regridder_weights' / f"weights_cmip_{model_name}_pr_conservative.nc"
            pr_regridder = alib.get_regridder(pr_data, "conservative", pr_w_file)
            pr_land = pr_regridder(pr_data[pr_var]).where(land_mask)
            
            # SST -> Bilinear
            tos_w_file = config.OUTPUT_DIR / 'regridder_weights' / f"weights_cmip_{model_name}_tos_bilinear.nc"
            tos_regridder = alib.get_regridder(tos_data, "bilinear", tos_w_file)
            tos_ocean = tos_regridder(tos_data[tos_var]).where(~land_mask)

            # --- C. Convert Units ---
            pr_land = alib.convert_precip_units(pr_land).persist()
            tos_ocean = tos_ocean.persist()

            # --- D. Run Centralized Analysis Loop ---
            alib.run_core_analysis_loop(
                pr_data=pr_land, 
                sst_data_for_indices=tos_ocean, 
                output_prefix=f"cmip_{model_name}", 
                land_mask=land_mask
            )
            
            print(f"✅ Successfully finished {model_name}")
            successful_models.append(model_name)

        except Exception as e:
            print(f"❌ ERROR processing {model_name}: {e}")

    print(f"\n--- 3. CMIP Analysis Complete ---")
    print(f"Successfully processed {len(successful_models)} models.")

if __name__ == "__main__":
    main()