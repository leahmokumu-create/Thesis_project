# analysis_lib.py
# It contains all the standardized helper functions for your project.
# Other scripts will import these functions (e.g., from analysis_lib import nan_detrend).
# --- Standard Library Imports ---
import sys
from collections import defaultdict
import numpy as np
import xarray as xr
import dask.array as da
import pandas as pd
import xesmf as xe
from scipy import stats
from scipy.signal import detrend
from scipy.stats import t as t_dist
from statsmodels.stats.multitest import fdrcorrection
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Rectangle, Patch
import config # <-- This is how we get all the settings

# =============================================================================
# 1. DATA PRE-PROCESSING FUNCTIONS
# =============================================================================

def align_calendar(ds):
    """
    Converts cftime.Datetime objects in a dataset to standard 
    np.datetime64[ns] for easier slicing and grouping.
    If conversion fails, it prints a warning and proceeds.
    """
    if 'time' in ds.coords and 'cftime' in str(ds.time.dtype):
        try:
            ds['time'] = ds.time.dt.to_datetimeindex(unsafe=True)
            print(f"    -> Aligned calendar for 'time' coordinate.")
        except Exception as e:
            print(f"    ⚠️ WARNING: Could not align calendar: {e}. Proceeding with cftime object.")
    return ds

def standardize_coords(ds):
    """
    Standardizes coordinate names and values for compatibility.
    - Renames 'latitude'/'Longitude' to 'lat'
    - Renames 'longitude'/'Longitude' to 'lon'
    - Converts longitude from 0-360 to -180 to 180 (if max > 180)
    - Sorts by 'lat' and 'lon'
    """
    print("    -> Standardizing coordinates...")
    coords_to_drop = []
    rename_dict = {}

    # --- Handle 'lat' ---
    if 'lat' in ds.coords:
        # 'lat' already exists. We should drop any conflicting names.
        if 'latitude' in ds.coords:
            coords_to_drop.append('latitude')
        if 'Latitude' in ds.coords:
            coords_to_drop.append('Latitude')
    else:
        # 'lat' does NOT exist. We must find a replacement.
        if 'latitude' in ds.coords:
            rename_dict['latitude'] = 'lat'
        elif 'Latitude' in ds.coords:
            rename_dict['Latitude'] = 'lat'

    # --- Handle 'lon' ---
    if 'lon' in ds.coords:
        # 'lon' already exists. Drop conflicts.
        if 'longitude' in ds.coords:
            coords_to_drop.append('longitude')
        if 'Longitude' in ds.coords:
            coords_to_drop.append('Longitude')
    else:
        # 'lon' does NOT exist. Find replacement.
        if 'longitude' in ds.coords:
            rename_dict['longitude'] = 'lon'
        elif 'Longitude' in ds.coords:
            rename_dict['Longitude'] = 'lon'

    # --- Perform the operations ---
    if coords_to_drop:
        actual_drops = [v for v in coords_to_drop if v in ds.coords]
        if actual_drops:
            ds = ds.drop_vars(actual_drops)
            print(f"       - Dropped conflicting coords: {actual_drops}")
    
    if rename_dict:
        actual_renames = {k: v for k, v in rename_dict.items() if k in ds.coords}
        if actual_renames:
            ds = ds.rename(actual_renames)
            print(f"       - Renamed: {actual_renames}")
   
    if 'lon' in ds.coords:
        try:
            # Check if longitude needs conversion
            lon_max = ds.lon.max().values
            if lon_max > 180:
                ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180))
                print(f"       - Converted lon from 0-360 to -180-180")
        except Exception as e:
                print(f"       - Could not check/convert lon: {e}")
    
    # Sort coordinates IF THEY ARE 1D DIMENSIONS
    if 'lat' in ds.coords and ds['lat'].ndim == 1:
        ds = ds.sortby('lat')
        print("       - Sorted by 1D 'lat' dim")
    if 'lon' in ds.coords and ds['lon'].ndim == 1:
        ds = ds.sortby('lon')
        print("       - Sorted by 1D 'lon' dim")
            
    return ds

def get_regridder(ds_in, method="bilinear", weight_filename=None):
    """
    Creates an xesmf regridder to the target grid, with robust error handling
    and weight caching.
    
    Args:
        ds_in (xr.Dataset or xr.DataArray): Input dataset with source grid.
        method (str): Regridding method ("bilinear" or "conservative").
        weight_filename (pathlib.Path, optional): Path to save/load weights.
    """
    
    # --- 1. Check for 1D Unstructured "Memory Bombs" ---
    # AWI models have giant 1D dims that will crash np.meshgrid
    SANITY_THRESHOLD = 5000 
    if 'lat' in ds_in.dims and ds_in.lat.ndim == 1 and len(ds_in.lat) > SANITY_THRESHOLD:
        raise ValueError(
            f"'lat' grid is a 1D unstructured mesh ({len(ds_in.lat)} elements). "
            "This is too large to regrid and would cause a memory error."
        )
    if 'lon' in ds_in.dims and ds_in.lon.ndim == 1 and len(ds_in.lon) > SANITY_THRESHOLD:
        raise ValueError(
            f"'lon' grid is a 1D unstructured mesh ({len(ds_in.lon)} elements). "
            "This is too large to regrid and would cause a memory error."
        )
        
    # --- 2. Clean "Dirty" Latitude Coordinates ---
    if 'lat' in ds_in.coords:
        ds_in['lat'] = ds_in['lat'].clip(-90.0, 90.0)
        print("       - Clipped 'lat' to [-90, 90] for regridder.")

    
    # --- 3. Check for existing weights file ---
    if weight_filename and weight_filename.exists():
        print(f"    -> Reusing {method} weights from: {weight_filename.name}")
        return xe.Regridder(
            ds_in, 
            config.DS_TARGET, 
            method, 
            reuse_weights=True, # <-- Load the weights
            filename=str(weight_filename),
            unmapped_to_nan=True, 
            ignore_degenerate=True
        )
    # --- 4. Create new weights ---
    else:
        print(f"    -> Creating {method} regridder to 1x1 grid...")
        if weight_filename:
            weight_filename.parent.mkdir(parents=True, exist_ok=True)
            print(f"       (Saving weights to: {weight_filename.name})")
            
        return xe.Regridder(
            ds_in, 
            config.DS_TARGET, 
            method, 
            reuse_weights=False, # <-- Create the weights
            filename=str(weight_filename) if weight_filename else None,
            unmapped_to_nan=True, 
            ignore_degenerate=True
        )
# =============================================================================
#  DATA LOADING & PREP HELPERS
# =============================================================================

def get_universal_land_mask():
    """
    Loads HadISST data *only* to create a standardized 1x1 land mask.
    This ensures all scripts use the exact same mask.
    Returns:
        land_mask_regridded (xr.DataArray): Mask (True for LAND) on the 1x1 grid.
    """
    print("--- (Helper) Loading HadISST to create universal land mask ---")
    try:
        time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
        sst_path = str(config.SST_PATH)
        
        with xr.open_mfdataset(sst_path, combine='by_coords', decode_times=time_coder) as ds_sst_raw:
            sst_std = standardize_coords(align_calendar(ds_sst_raw))
            
            # Find the sst variable
            if 'sst' in sst_std.data_vars:
                sst_var_name = 'sst'
            else:
                possible_vars = [v for v in sst_std.data_vars if v not in sst_std.dims and 'bnds' not in v]
                if not possible_vars:
                    raise ValueError("Could not find a valid 'sst' data variable in HadISST file.")
                sst_var_name = possible_vars[0]
            
            # Create Land Mask (True for LAND)
            print("  -> Creating land-sea mask from source grid...")
            land_mask_obs = sst_std[sst_var_name].isel(time=0).isnull()
            mask_regridder = get_regridder(land_mask_obs, method="bilinear")
            land_mask_regridded = (mask_regridder(land_mask_obs) > 0.5).persist() # True for LAND

            print("✅ Universal land mask (1x1) is ready.")
            return land_mask_regridded

    except Exception as e:
        print(f"❌ CRITICAL ERROR: Failed to create land mask. Error: {e}")
        sys.exit()

def load_processed_obs(load_full_range=True):
    """
    Loads HadISST and GPCC, standardizes, regrids, and converts units.
    """
    print(f"--- (Lib) Loading Observations (Full Range={load_full_range}) ---")
    
    # Define time slice
    if load_full_range:
        t_slice = slice(str(config.TS_START_YEAR), str(config.TS_END_YEAR))
    else:
        t_slice = slice(str(config.START_YEAR), str(config.END_YEAR))

    # 1. Universal Land Mask
    land_mask = get_universal_land_mask()
    
    # 2. Load HadISST
    WEIGHT_DIR = config.OUTPUT_DIR / 'regridder_weights'
    with xr.open_mfdataset(config.SST_PATH, combine='by_coords', use_cftime=True) as ds:
        sst_data = standardize_coords(align_calendar(ds)).sel(time=t_slice)
        sst_var = [v for v in sst_data.data_vars if v not in sst_data.dims][0]
    
    sst_w_file = WEIGHT_DIR / "weights_obs_hadisst_bilinear.nc"
    sst_regridder = get_regridder(sst_data, "bilinear", sst_w_file)
    sst_ocean = sst_regridder(sst_data[sst_var]).where(~land_mask).persist()

    # 3. Load GPCC
    with xr.open_mfdataset(config.GPCC_PATH, combine='by_coords', use_cftime=True) as ds:
        pr_data = standardize_coords(align_calendar(ds)).sel(time=t_slice)
        pr_var = [v for v in pr_data.data_vars if v not in pr_data.dims][0]

    pr_w_file = WEIGHT_DIR / "weights_obs_gpcc_conservative.nc"
    pr_regridder = get_regridder(pr_data, "conservative", pr_w_file)
    pr_land = pr_regridder(pr_data[pr_var]).where(land_mask)
    pr_land = convert_precip_units(pr_land).persist()

    return land_mask, sst_ocean, pr_land
# =============================================================================
# 2. CORE STATISTICAL & ANALYSIS FUNCTIONS
# =============================================================================
def calculate_anomalies(da, dim='time'):
    """
    Calculates monthly anomalies (removes climatological seasonal cycle).
    """
    return da.groupby(f'{dim}.month') - da.groupby(f'{dim}.month').mean(dim)

def nan_detrend(y):
    """
    Detrends a 1D array using scipy.signal.detrend, ignoring NaNs.
    This is the function applied by apply_ufunc.
    """
    not_nan = ~np.isnan(y)
    if np.sum(not_nan) < 2: # Need at least 2 points to detrend
        return y
    y_detrended = np.full_like(y, np.nan)
    y_detrended[not_nan] = detrend(y[not_nan])
    return y_detrended

def detrend_dim(da_in, dim='time'):
    """
    Applies nan_detrend to every grid cell along a specified dimension.
    """
    return xr.apply_ufunc(
        nan_detrend, da_in,
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
        dask='parallelized',
        output_dtypes=[da_in.dtype],
        vectorize=True
    )

def corr_and_p_value_autocorr_corrected(x, y):
    """
    Calculates Spearman (rank) correlation and p-value, 
    correcting for autocorrelation (Neff).
    """
    valid_mask = ~np.isnan(x) & ~np.isnan(y)
    
    # --- Convert raw values to ranks ---
    x_ranked = stats.rankdata(x[valid_mask])
    y_ranked = stats.rankdata(y[valid_mask])
    N = len(x_ranked)

    if N < 4: 
        return np.nan, np.nan

    # Compute Pearson correlation *on the ranks* (this is Spearman's rho)
    r = np.corrcoef(x_ranked, y_ranked)[0, 1]

    # Compute autocorrelation *of the ranks*
    r1 = np.corrcoef(x_ranked[:-1], x_ranked[1:])[0, 1]
    r2 = np.corrcoef(y_ranked[:-1], y_ranked[1:])[0, 1]

    if np.isnan(r1) or np.isnan(r2): 
        return r, np.nan  # Return correlation but no valid p-value
 
    N_eff = N * (1 - r1 * r2) / (1 + r1 * r2)
    df_eff = N_eff - 2

    if df_eff <= 1 or np.isnan(df_eff): 
        return r, np.nan
    if abs(r) >= 1.0: 
        return r, 0.0  # Perfect correlation

    t_stat = r * np.sqrt(df_eff / (1 - r**2))
    p_value = t_dist.sf(np.abs(t_stat), df_eff) * 2

    return r, p_value

def calculate_slope_and_pvalue(y):
    """
    Calculates the (robust) Theil-Sen slope and Mann-Kendall p-value 
    for a 1D time series.
    """
    x = np.arange(len(y))
    finite_mask = np.isfinite(y)
    x_valid = x[finite_mask]
    y_valid = y[finite_mask]
    
    if len(y_valid) < 3: 
        return np.nan, np.nan
        
    try:
        slope, _, _, _ = stats.theilslopes(y_valid, x_valid, alpha=1-config.SIGNIFICANCE_LEVEL)
        _, p_value = stats.kendalltau(x_valid, y_valid)
        return slope, p_value
    except (ValueError, IndexError):
        return np.nan, np.nan
#not relevant for my study
def calculate_fdr_mask(p_value_map, alpha=None):
    """
    Performs a False Discovery Rate (FDR) correction on a 2D p-value map.
    Returns a 2D boolean mask of significant pixels.
    """
    if alpha is None:
        alpha = config.SIGNIFICANCE_LEVEL

    p_values_flat = p_value_map.data.flatten()
    valid_indices = ~np.isnan(p_values_flat)
    p_values_valid = p_values_flat[valid_indices]
    
    if len(p_values_valid) == 0:
        print("    -> No valid p-values found for FDR correction.")
        return xr.full_like(p_value_map, False, dtype=bool)

    # Run the FDR correction
    rejected, _ = fdrcorrection(p_values_valid, alpha=alpha, method='indep')
    
    # Reshape the boolean mask back to 2D
    mask_flat = np.full(p_values_flat.shape, False)
    mask_flat[valid_indices] = rejected
    fdr_mask_2d = mask_flat.reshape(p_value_map.shape)
    
    return xr.DataArray(
        fdr_mask_2d,
        coords=p_value_map.coords,
        dims=p_value_map.dims,
        name='fdr_significant'
    )

def get_region_mean(data_in, region_name):
    """
    Helper function to get a spatial mean from a region in config.BOUNDS.
    Handles pacific-crossing regions.
    """
    b = config.BOUNDS[region_name]
    
    # Check if region is pacific-crossing
    is_pacific = b['lon_min'] > b['lon_max']
    
    if is_pacific:
        part1 = data_in.sel(lat=slice(b['lat_min'], b['lat_max']), lon=slice(b['lon_min'], 180))
        part2 = data_in.sel(lat=slice(b['lat_min'], b['lat_max']), lon=slice(-180, b['lon_max']))
        region_data = xr.concat([part1, part2], dim='lon')
    else:
        region_data = data_in.sel(lat=slice(b['lat_min'], b['lat_max']), lon=slice(b['lon_min'], b['lon_max']))
        
    return region_data.mean(dim=['lat', 'lon'])

def compute_indices(sst_da):
    """
    Computes all SST indices based on the definitions in config.py.
    """
    indices = {}
    region_cache = {}
    
    def get_mean(region_name):
        if region_name not in region_cache:
            region_cache[region_name] = get_region_mean(sst_da, region_name)
        return region_cache[region_name]

    for name, logic in config.INDEX_DEFINITIONS.items():
        op_type = logic[0]
        
        if op_type == 'mean':
            indices[name] = get_mean(logic[1])
        elif op_type == 'subtract':
            term1 = get_mean(logic[1])
            term2 = get_mean(logic[2])
            indices[name] = term1 - term2
             
    return indices


# =============================================================================
# 4. PLOTTING FUNCTIONS
# =============================================================================
# These functions are used by make_figures.py to create the final plots.


def add_significance(ax, p_value_map, fdr_mask=None, significance_level=None):
    """
    Helper function to add stippling for significance to a map.
    - Sparse dots (.) for simple p-value
    - Dense dots (...) for FDR-corrected
    """
    if significance_level is None:
        significance_level = config.SIGNIFICANCE_LEVEL

    # --- Plot 1: Simple Significance (p < level) ---
    simple_mask = p_value_map < significance_level
    if simple_mask.any():
        print(f"    -> Plotting simple significance (p < {significance_level})")
        simple_mask_bool = simple_mask.astype(bool).where(~np.isnan(p_value_map))
        ax.contourf(
            simple_mask_bool.lon, simple_mask_bool.lat, simple_mask_bool,
            levels=[0.5, 1.5], colors='none', hatches=['.'],
            edgecolors='gray', transform=ccrs.PlateCarree(), zorder=3
        )

    # --- Plot 2: FDR Significance (q < level) ---
    if fdr_mask is not None and fdr_mask.any():
        print(f"    -> Plotting FDR significance (q < {significance_level})")
        fdr_mask_bool = fdr_mask.astype(bool).where(~np.isnan(p_value_map))
        ax.contourf(
            fdr_mask_bool.lon, fdr_mask_bool.lat, fdr_mask_bool,
            levels=[0.5, 1.5], colors='none', hatches=['...'],
            edgecolors='black', transform=ccrs.PlateCarree(), zorder=4
        )

def add_ensemble_significance(ax, p_value_map, agreement_map=None, significance_level=None):
    """
    Helper function to add significance stippling for ENSEMBLE means.
    - Stippling (...) for statistical significance (t-test p-value)
    - (Optional) Contours for model agreement
    """
    if significance_level is None:
        significance_level = config.SIGNIFICANCE_LEVEL
        
    # --- Plot T-Test Significance (p < level) ---
    sig_mask = p_value_map < significance_level
    if sig_mask.any():
        print(f"    -> Plotting ensemble significance (p < {significance_level})")
        sig_mask_bool = sig_mask.astype(bool).where(~np.isnan(p_value_map))
        ax.contourf(
            sig_mask_bool.lon, sig_mask_bool.lat, sig_mask_bool,
            levels=[0.5, 1.5], colors='none', hatches=['...'],
            edgecolors='black', transform=ccrs.PlateCarree(), zorder=4
        )

def create_map_plot(ax, data_map, title, cbar_label, extent, boxes_to_add, add_colorbar=True):
    """
    Core plotting function that draws a single map on a given axis.
    """
    if 'Trend' in cbar_label:
        vmax = 9.0  
        vmin = -vmax
        levels = np.linspace(vmin, vmax, 19) 
        cmap = 'BrBG'
    else: 
        vmin, vmax = -0.8, 0.8
        levels = np.linspace(vmin, vmax, 17)
        cmap = 'seismic_r'
    
    ax.add_feature(cfeature.LAND, zorder=0, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, zorder=0, facecolor='lightblue')
    
    im = None
    
    if not data_map.isnull().all():
        if add_colorbar:
            im = data_map.plot.contourf(
                ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, levels=levels,
                cbar_kwargs={'label': cbar_label, 'orientation': 'horizontal', 'pad': 0.1, 'shrink': 0.7},
                vmin=vmin, vmax=vmax, extend='both', zorder=2
            )
        else:
            im = data_map.plot.contourf(
                ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, levels=levels,
                add_colorbar=False, 
                vmin=vmin, vmax=vmax, extend='both', zorder=2
            )
    
    ax.add_feature(cfeature.COASTLINE, edgecolor='black', zorder=5)
    ax.add_feature(cfeature.BORDERS, linestyle=':', zorder=5)
    
    gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels, gl.right_labels = False, False
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.set_title(title, fontsize=12)
    
    for box_name in boxes_to_add:
        b = config.BOUNDS[box_name]
        box_color = 'green' if box_name == 'study' else 'purple'
        rect = Rectangle(
            (b['lon_min'], b['lat_min']), b['lon_max'] - b['lon_min'], b['lat_max'] - b['lat_min'],
            facecolor='none', transform=ccrs.PlateCarree(), zorder=6,
            edgecolor=box_color, linewidth=2, linestyle='-' 
        )
        ax.add_patch(rect)

    legend_handles = []
    if 'study' in boxes_to_add:
        legend_handles.append(Patch(facecolor='none', edgecolor='green', linewidth=2, linestyle='-', label='Study Area'))
    if 'ea' in boxes_to_add:
        legend_handles.append(Patch(facecolor='none', edgecolor='purple', linewidth=2, linestyle='-', label='East Africa'))
    
    if legend_handles:
        ax.legend(handles=legend_handles, loc='lower left', fontsize=10)

    return im

def create_map_plot_teleconnection(ax, data_map, title, cbar_label, extent, boxes_to_add, add_colorbar=True):
    """
    Specialized plotting function for the Teleconnection map
    (Precip Index vs. SST Field) that draws all the colored boxes and text.
    
    """
    # --- Colorbar ---
    vmin, vmax = -0.8, 0.8 # Use -0.8 to 0.8 for correlations
    levels = np.linspace(vmin, vmax, 17) # 17 levels
    cmap = 'seismic_r'
    
    # --- Draw Map Features ---
    ax.add_feature(cfeature.LAND, zorder=1, facecolor='lightgray')
    ax.add_feature(cfeature.COASTLINE, zorder=4)
    
    im = None
    
    # --- Plot Data ---
    plot_kwargs = {
        'ax': ax, 'transform': ccrs.PlateCarree(), 'cmap': cmap, 
        'levels': levels, 'vmin': vmin, 'vmax': vmax, 
        'extend': 'both', 'zorder': 2
    }
    
    if add_colorbar:
        plot_kwargs['cbar_kwargs'] = {'label': cbar_label, 'orientation': 'horizontal', 'pad': 0.1, 'shrink': 0.5}
    else:
        plot_kwargs['add_colorbar'] = False

    im = None
    
    # --- Plot Data ---
    if not data_map.isnull().all():
        if add_colorbar:
            im = data_map.plot.contourf(
                ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, levels=levels,
                cbar_kwargs={'label': cbar_label, 'orientation': 'horizontal', 'pad': 0.1, 'shrink': 0.5},
                vmin=vmin, vmax=vmax, extend='both', zorder=2
            )
        else:
            im = data_map.plot.contourf(
                ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, levels=levels,
                add_colorbar=False, # <--- THIS KILLS THE INDIVIDUAL COLORBAR
                vmin=vmin, vmax=vmax, extend='both', zorder=2
            )
    
    # --- Draw Overlays ---
    ax.set_global()
    gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels, gl.right_labels = False, False
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.set_title(title, fontsize=12)
    
    # --- Add Region Boxes (with custom colors and text) ---
    text_style_common = {
        'fontsize': 9, 'color': 'black', 'weight': 'bold',
        'bbox': {'facecolor': 'white', 'alpha': 0.6, 'edgecolor': 'none', 'boxstyle': 'round,pad=0.2'}
    }

    # Define the colors for each box
    box_styles = {
        'study':    {'color': 'red', 'style': {'linewidth': 3, 'linestyle': '-'}},
        'indopacific': {'color': 'black', 'style': {'linestyle': '--', 'linewidth': 2}},
        'wpo':      {'color': 'black', 'style': {'linestyle': '--', 'linewidth': 2}},
       # 'PDO':      {'color': 'brown', 'style': {'linestyle': '--', 'linewidth': 2}}, # <-- Includes PDO fix
        'nino12':   {'color': 'darkorange', 'style': {'linestyle': '--', 'linewidth': 2}},
        'nino3':    {'color': 'brown', 'style': {'linestyle': '--', 'linewidth': 2}},
        'nino34':   {'color': 'magenta', 'style': {'linestyle': '--', 'linewidth': 2}},
        'nino4':    {'color': 'green', 'style': {'linestyle': '--', 'linewidth': 2}},
        'wio':      {'color': 'purple', 'style': {'linestyle': '--', 'linewidth': 2}},
        'eio':      {'color': 'purple', 'style': {'linestyle': '--', 'linewidth': 2}},
       
    }

    for box_name in boxes_to_add:
        if box_name not in config.BOUNDS:
            continue
        
        b = config.BOUNDS[box_name]
        lon_min, lon_max, lat_min, lat_max = b['lon_min'], b['lon_max'], b['lat_min'], b['lat_max']
        is_pacific = lon_min > lon_max
        
        # Get the style, default to black
        style_info = box_styles.get(box_name, {'color': 'black', 'style': {'linestyle': '--'}})
        style = {'edgecolor': style_info['color'], 'facecolor': 'none', 'transform': ccrs.PlateCarree(), 'zorder': 10}
        style.update(style_info['style']) # Add lw, linestyle, etc.

        if is_pacific: # Pacific-crossing
            ax.add_patch(Rectangle((lon_min, lat_min), 180 - lon_min, lat_max - lat_min, **style))
            ax.add_patch(Rectangle((-180, lat_min), 180 + lon_max, lat_max - lat_min, **style))
        else:
            ax.add_patch(Rectangle((lon_min, lat_min), lon_max - lon_min, lat_max - lat_min, **style))
        
  # --- Add Legend for Region Boxes ---
    legend_handles = []

    # Map internal names to display names for the legend
    label_map = {
        'study': 'Study Area',
        'indopacific': 'Indo-Pacific',
        'wpo': 'WPO',
        'nino12': 'Niño 1+2',
        'nino3': 'Niño 3',
        'nino34': 'Niño 3.4',
        'nino4': 'Niño 4',
        'wio': 'IOD West',
        'eio': 'IOD East', # From your target image
    }

    # Dynamically create a legend item for each box
    for box_name in boxes_to_add:
        if box_name in label_map and box_name in box_styles:
            style_info = box_styles[box_name]
            label = label_map[box_name]
            
            legend_handles.append(Patch(facecolor='none', 
                                        edgecolor=style_info['color'], 
                                        linewidth=style_info['style'].get('linewidth', 2), 
                                        linestyle=style_info['style'].get('linestyle', '-'),
                                        label=label))
    
   # 2. Add Significance Stippling Entry manually
    # '...' is the hatch code for dots
    legend_handles.append(Patch(facecolor='none', edgecolor='black', hatch='...', label='Significance (95%)'))
    
    # Return the handles so the main script can plot them globally
    return im, legend_handles

def force_time_to_years(ts_data):
    """
    Robustly converts any xarray DataArray time axis into a simple
    numpy array of integer years for plotting.
    This handles:
    - Standard datetime objects
    - cftime objects (noleap, 360_day, etc.)
    - Integer year axes (which are returned as-is)
    - Generic 'object' axes (which we assume are cftime)
    """
    
    # 1. Check if time is already a simple integer year (like for 'MAM' plots)
    if np.issubdtype(ts_data.time.dtype, np.integer):
        return ts_data # It's already [1981, 1982...], do nothing.

    # 2. Check if it's a standard datetime
    if np.issubdtype(ts_data.time.dtype, np.datetime64):
        ts_data['time'] = ts_data.time.dt.year
        return ts_data
    # 3. Check for cftime or generic object dtype
    if 'cftime' in str(ts_data.time.dtype) or ts_data.time.dtype == 'object':
        try:
            # Manually extract the 'year' from each object
            plottable_years = [date.year for date in ts_data.time.values]
            ts_data['time'] = plottable_years
            return ts_data
        except Exception as e:
            # This will be caught by the 'try...except' in make_figures.py
            print(f"    [!] Warning: Could not extract year from cftime/object dtype: {e}")
            raise # Re-raise the error
            
    # 4. If we still don't know what it is, raise the error.
    raise TypeError(f"Unknown time coordinate type: {ts_data.time.dtype}")

def convert_precip_units(pr_da):
    """
    Converts precipitation units based STRICTLY on metadata.
    Does not guess based on magnitude (Physical Causality).
    """
    # 1. Extract Unit String
    units = pr_da.attrs.get('units', '').strip().lower()
    
    # Define conversion factor: kg m-2 s-1 to mm/month
    # Average seconds in a Gregorian month (365.2425 days / 12) * 24 * 60 * 60
    # or simple 30-day approximation depending on your calendar.
    # CMIP standard is usually kg m-2 s-1.
    SECONDS_PER_MONTH = 2629746.0 

    print(f"       [Unit Check] Metadata says: '{units}'")

    # 2. Case A: Flux Units (Standard CMIP/AMIP)
    # Common variations: "kg m-2 s-1", "kg/m^2/s", "kg m**-2 s**-1"
    if 'kg' in units and 's' in units:
        print("       -> Detected Flux (kg m-2 s-1). Converting to mm/month...")
        pr_converted = pr_da * SECONDS_PER_MONTH
        pr_converted.attrs['units'] = 'mm/month'
        return pr_converted

    # 3. Case B: Meters (Some raw models)
    elif units == 'm':
        print("       -> Detected Meters (m). Converting to mm/month...")
        pr_converted = pr_da * 1000.0
        pr_converted.attrs['units'] = 'mm/month'
        return pr_converted

    # 4. Case C: Already Millimeters (GPCC, CRU, etc.)
    # Common variations: "mm", "mm/month", "millimeter"
    elif 'mm' in units:
        print("       -> Detected mm. Keeping as is.")
        return pr_da

    # 5. Case D: The "Physical Causality" Safety Net
    # If we don't know the units, we DO NOT GUESS. We raise an error.
    else:
        # Check if the user passed data with no units at all
        if not units:
             raise ValueError(
                "❌ CRITICAL ERROR: Variable has no 'units' attribute. "
                "Cannot proceed safely. Please assign units to your DataArray "
                "before processing (e.g., da.attrs['units'] = 'mm/month')."
            )
        else:
            raise ValueError(
                f"❌ CRITICAL ERROR: Unknown units '{units}'. "
                "Script only handles 'kg m-2 s-1', 'm', or 'mm'. "
                "Please convert manually."
            )
def get_period_data(data_array, period_name, months, is_anomaly=False):
    """
    Selects data for a given period and calculates the correct
    time series for trend (raw) or teleconnections (anomalies).
    
    This is the function that fixes BOTH of your plotting bugs.
    
    - For 'MAM', returns a yearly mean.
    - For 'April', etc.:
        - if is_anomaly=True (for Teleconnections), returns a yearly mean.
          (This fixes the "flatline" spaghetti plot).
        - if is_anomaly=False (for Trend), returns the raw monthly data.
          (This fixes the "skewed" histogram).
    """
    if period_name == 'MAM':
        # MAM is always a yearly mean
        period_data = data_array.sel(time=data_array['time.month'].isin(months)).groupby('time.year').mean('time').rename({'year': 'time'})
    else:
        # For monthly periods, the logic is different
        if is_anomaly:
            # For teleconnections/regression, we want a single yearly value
            # This fixes the "flatline" spaghetti plot
            period_data = data_array.sel(time=data_array['time.month'].isin(months)).groupby('time.year').mean('time').rename({'year': 'time'})
        else:
            # For trend, we want the raw monthly data (e.g., all Aprils)
            # This fixes the "skewed" histogram
            period_data = data_array.sel(time=data_array['time.month'].isin(months))
            
    return period_data.chunk(dict(time=-1))

# =============================================================================
# 7. ANALYSIS HELPER FUNCTIONS (continued)
# =============================================================================

def smooth_trend_map(slope_map_computed):
    """
    Applies standard unit conversion and smoothing to a trend map.
    - Converts slope/year to (mm/month)/decade.
    - Applies a 5x5 rolling mean.
    - Re-masks the smoothed data.
    """
    print("       - Converting trend to (mm/month)/decade and smoothing...")
    trend_map_per_decade = slope_map_computed * 10
    trend_map_smoothed = trend_map_per_decade.rolling(lat=5, lon=5, center=True, min_periods=1).mean()
    
    # Re-masking is critical
    trend_map_smoothed = trend_map_smoothed.where(trend_map_per_decade.notnull())
    
    return trend_map_smoothed



# analysis_lib.py

def run_core_analysis_loop(pr_data, sst_data_for_indices, output_prefix, land_mask):
    """
    Performs the analysis.
    - Uses FULL data (1950-2020) for Regression Time Series (Step C).
    - Uses CORE data (1981-2014) for Trend & Correlation Maps (Step A, D, E).
    - Calculates a scientifically valid Global Warming (GW) Proxy.
    """
    print(f"--- Running Analysis Loop for: {output_prefix} ---")

    # 1. Calculate Anomalies on the FULL dataset (e.g. 1950-2020)
    # Note: This sets the 'Zero' line based on the full available period.
    pr_anom_full = calculate_anomalies(pr_data)
    sst_anom_full = calculate_anomalies(sst_data_for_indices)

    # We use latitude weighting to ensure the poles aren't over-represented.
    # This creates a physically valid "Global Warming" time series.
    weights = np.cos(np.deg2rad(sst_anom_full.lat))
    gw_proxy_full = sst_anom_full.weighted(weights).mean(dim=['lat', 'lon'])
    # -------------------------------------------------------------

    for period_name, months in config.ANALYSIS_PERIODS.items():
        print(f"   -> Period: {period_name}")

        # =====================================================================
        # PREP STEP: CREATE TWO DATA STREAMS
        # =====================================================================

        # STREAM 1: FULL (1950-2020) -> For Time Series Plots
        pr_anom_yr_full = get_period_data(pr_anom_full, period_name, months, is_anomaly=True)
        sst_anom_yr_full = get_period_data(sst_anom_full, period_name, months, is_anomaly=True)
        gw_proxy_yr_full = get_period_data(gw_proxy_full, period_name, months, is_anomaly=True) # <--- New

        # STREAM 2: CORE (1981-2014) -> For Maps
        # We slice the FULL anomalies down to the CORE period
        core_slice = slice(str(config.START_YEAR), str(config.END_YEAR))

        pr_anom_yr_core = pr_anom_yr_full.sel(time=core_slice)
        sst_anom_yr_core = sst_anom_yr_full.sel(time=core_slice)

        # We also need raw data slice for the Trend Map
        pr_raw_core = pr_data.sel(time=core_slice)
        pr_trend_data_core = get_period_data(pr_raw_core, period_name, months, is_anomaly=False)

        # =====================================================================
        # ANALYSIS A: TREND MAP (Uses CORE 1981-2014)
        # =====================================================================
        slope, pval = xr.apply_ufunc(
            calculate_slope_and_pvalue, pr_trend_data_core,
            input_core_dims=[['time']], output_core_dims=[[], []],
            exclude_dims=set(('time',)), dask="parallelized", vectorize=True,
            output_dtypes=[pr_trend_data_core.dtype, np.float64]
        )
        slope_c, pval_c = da.compute(slope, pval)
        trend_smooth = smooth_trend_map(slope_c)

        trend_smooth.rename("precip_trend").to_netcdf(config.OUTPUT_DIR / f"{output_prefix}_trend_{period_name.lower()}.nc")
        pval_c.rename("p_value").to_netcdf(config.OUTPUT_DIR / f"{output_prefix}_trend_pval_{period_name.lower()}.nc")

        # =====================================================================
        # ANALYSIS B: DETRENDING & INDICES
        # =====================================================================

        # 1. Detrend FULL (For Time Series)
        pr_detr_full = detrend_dim(pr_anom_yr_full, 'time')
        sst_detr_full = detrend_dim(sst_anom_yr_full, 'time')

        sst_idx_full = compute_indices(sst_detr_full)
        pr_idx_full = get_region_mean(pr_detr_full, 'study').chunk(dict(time=-1))

        # 2. Detrend CORE (For Maps)
        pr_detr_core = detrend_dim(pr_anom_yr_core, 'time')
        sst_detr_core = detrend_dim(sst_anom_yr_core, 'time')

        sst_idx_core = compute_indices(sst_detr_core)
        pr_idx_core = get_region_mean(pr_detr_core, 'study').chunk(dict(time=-1))

        # =====================================================================
        # ANALYSIS C: SAVE TIME SERIES (Raw Anomalies & Detrended)
        # =====================================================================

        # --- 1. Compute Raw Indices (Trend Preserved) ---
        # Critical for Trend Attribution plots and Plume plots
        sst_idx_full_raw = compute_indices(sst_anom_yr_full)
        pr_idx_full_raw = get_region_mean(pr_anom_yr_full, 'study').chunk(dict(time=-1))

        reg_data_raw = {
            'pr_index_study': pr_idx_full_raw.rename('pr_index_study'),
            'GW_Proxy': gw_proxy_yr_full.rename('GW_Proxy')  # <--- SAVE IT HERE
        }
        for k, v in sst_idx_full_raw.items():
            if k in config.INDICES_TO_PLOT:
                reg_data_raw[k] = v.rename(k)

        # Save RAW file
        xr.Dataset(reg_data_raw).to_netcdf(
            config.OUTPUT_DIR / f"{output_prefix}_regression_timeseries_{period_name.lower()}.nc"
        )

        # --- 2. Compute Detrended Indices (Trend Removed) ---
        # Useful for checking pure internal variability correlations
        sst_idx_full_detr = compute_indices(sst_detr_full)
        pr_idx_full_detr = get_region_mean(pr_detr_full, 'study').chunk(dict(time=-1))

        reg_data_detr = {'pr_index_study': pr_idx_full_detr.rename('pr_index_study')}
        for k, v in sst_idx_full_detr.items():
            if k in config.INDICES_TO_PLOT:
                reg_data_detr[k] = v.rename(k)

        # Save DETRENDED file
        xr.Dataset(reg_data_detr).to_netcdf(
            config.OUTPUT_DIR / f"{output_prefix}_regression_timeseries_detrended_{period_name.lower()}.nc"
        )

        # =====================================================================
        # ANALYSIS D: TELECONNECTIONS (Uses CORE 1981-2014)
        # =====================================================================

        # 1. PR Index (Core) vs SST Field (Core)
        corr, pval = xr.apply_ufunc(
            corr_and_p_value_autocorr_corrected, pr_idx_core, sst_detr_core,
            input_core_dims=[['time'], ['time']], output_core_dims=[[], []],
            exclude_dims=set(('time',)), vectorize=True, dask="parallelized"
        )
        c_comp, p_comp = da.compute(corr, pval)
        c_smooth = c_comp.rolling(lat=5, lon=5, center=True, min_periods=1).mean().where(c_comp.notnull())

        c_smooth.rename("correlation").to_netcdf(config.OUTPUT_DIR / f"{output_prefix}_tele_pr-index-vs-sst-field_{period_name.lower()}.nc")
        p_comp.rename("p_value").to_netcdf(config.OUTPUT_DIR / f"{output_prefix}_tele_pr-index-vs-sst-field_pval_{period_name.lower()}.nc")

        # 2. SST Index (Core) vs PR Field (Core)
        for name in config.INDICES_TO_PLOT:
            if name not in sst_idx_core: continue
            idx_ts = sst_idx_core[name].chunk(dict(time=-1))
            corr, pval = xr.apply_ufunc(
                corr_and_p_value_autocorr_corrected, idx_ts, pr_detr_core,
                input_core_dims=[['time'], ['time']], output_core_dims=[[], []],
                exclude_dims=set(('time',)), vectorize=True, dask="parallelized"
            )
            c_comp, p_comp = da.compute(corr, pval)
            c_smooth = c_comp.rolling(lat=5, lon=5, center=True, min_periods=1).mean().where(land_mask).where(c_comp.notnull())

            c_smooth.rename("correlation").to_netcdf(config.OUTPUT_DIR / f"{output_prefix}_tele_sst-index-vs-pr-field_{name.lower()}_{period_name.lower()}.nc")
            p_comp.rename("p_value").to_netcdf(config.OUTPUT_DIR / f"{output_prefix}_tele_sst-index-vs-pr-field_pval_{name.lower()}_{period_name.lower()}.nc")

# =============================================================================
# 8. CENTRALIZED DATA LOADING FUNCTIONS
# =============================================================================

def fix_time_axis(da):
    """
    Standardizes time axis to integer years.
    Used for timeseries plots to ensure x-axis alignment.
    """
    if 'time' in da.dims and not np.issubdtype(da.time.dtype, np.integer):
        try:
            da['time'] = [t.year for t in da.time.values]
        except: pass
    return da

def load_ranking_data(period):
    """
    Loads data for 'cleanscript.py' and 'dash_clean2.py'.
    Focus: Individual model Timeseries and Trend Maps for ranking.
    Returns: Dictionary with lists of individual model data.
    """
    print(f"    ... (Lib) Loading Ranking/Timeseries data for: {period.upper()} ...")
    cache = {'obs': {'data': None, 'extras': {}, 'trend_map': None}, 'amip': [], 'cmip': []}
    
    # Internal helper to process a single regression file
    def _process_file(f, var_name):
        try:
            with xr.open_dataset(f) as ds:
                if var_name in ds:
                    da = ds[var_name].load()
                    da = fix_time_axis(da)
                    extras = {}
                    # Load indices defined in config
                    for idx in config.INDICES_TO_PLOT:
                        if idx in ds:
                            extras[idx] = fix_time_axis(ds[idx].load())
                    return da, extras
        except Exception as e:
            print(f"    [!] Error loading {f.name}: {e}")
        return None, None

    # 1. Load Observations
    obs_ts_file = config.OUTPUT_DIR / f"obs_regression_timeseries_{period.lower()}.nc"
    obs_trend_file = config.OUTPUT_DIR / f"obs_trend_{period.lower()}.nc"
    
    if obs_ts_file.exists():
        da, extras = _process_file(obs_ts_file, "pr_index_study")
        cache['obs']['data'] = da
        cache['obs']['extras'] = extras

    if obs_trend_file.exists():
         with xr.open_dataset(obs_trend_file) as ds:
             if "precip_trend" in ds: cache['obs']['trend_map'] = ds["precip_trend"].load()

    # 2. Load Models (Iterative approach for individual rankings)
    # Load trend maps first to link them to timeseries later
    trend_map_cache = {'amip': {}, 'cmip': {}}
    for m_type in ['amip', 'cmip']:
        for tf in config.OUTPUT_DIR.glob(f"{m_type}_*_{'trend'}_{period.lower()}.nc"):
            try:
                ds = xr.open_dataset(tf)
                if 'precip_trend' in ds:
                    trend_map_cache[m_type][tf.stem] = ds['precip_trend'].load()
            except: pass

    # Load Timeseries and link to Trend Maps
  # B. Load Timeseries and Link to Trend Maps
    ignore_list = [
        'amip', 'cmip', 'trend', 'mam', 'jja', 'son', 'djf', 
        'pr', 'precip', 'regression', 'timeseries',
        'amon', 'omon', 'day', 'aero', 'historical', 'picontrol',
        'r1i1p1', 'r1i1p1f1', 'gn', 'gr', 
        period.lower() 
    ]
    
    for model_type in ['amip', 'cmip']:
        ts_pattern = f"{model_type}_*_{'regression_timeseries'}_{period.lower()}.nc"
        ts_files = sorted(config.OUTPUT_DIR.glob(ts_pattern))
        
        for f in ts_files:
            if model_type == 'cmip' and "_r2_" in f.name: continue
            
            da, extras = _process_file(f, "pr_index_study")
            if da is not None:
                parts = f.stem.split('_')
                # Extract clean model name
                name = next((p for p in parts if p.lower() not in ignore_list), "Unknown")
                
                # --- FIXED EXACT MATCHING ---
                trend_da = None
                for k, v in trend_map_cache[model_type].items():
                    # k is the filename stem, e.g., "amip_CESM2-WACCM_trend_april"
                    # We split it to ensure we match "CESM2" against a distinct part
                    k_parts = k.split('_')
                    if name in k_parts: 
                        trend_da = v
                        break
                
                cache[model_type].append({
                    'name': name, 
                    'data': da, 
                    'extras': extras, 
                    'trend_map': trend_da, 
                    'file': f
                })
    return cache

def load_ensemble_maps(period):
    """
    Loads data for 'runscript.py'.
    Focus: Spatial Maps (Trend, Teleconnections) concatenated for Ensembles.
    Returns: Dictionary with 'obs', 'amip' (concat), 'cmip' (concat).
    """
    print(f"    ... (Lib) Loading Spatial/Ensemble data for: {period.upper()} ...")
    period_lc = period.lower()
    data = {'obs': {}, 'amip': {}, 'cmip': {}}

    # 1. Load Observations
    try:
        # Trend
        f_trend = config.OUTPUT_DIR / f"obs_trend_{period_lc}.nc"
        f_pval = config.OUTPUT_DIR / f"obs_trend_pval_{period_lc}.nc"
        if f_trend.exists(): 
            data['obs']['trend'] = xr.open_dataarray(f_trend)
            data['obs']['trend_pval'] = xr.open_dataarray(f_pval)
            data['obs']['trend_fdr'] = calculate_fdr_mask(data['obs']['trend_pval'])
        
        # Teleconnections
        for index in config.INDICES_TO_PLOT:
            idx_lc = index.lower()
            # PR Index vs SST Field
            f_sst = config.OUTPUT_DIR / f"obs_tele_pr-index-vs-sst-field_{period_lc}.nc"
            f_sst_p = config.OUTPUT_DIR / f"obs_tele_pr-index-vs-sst-field_pval_{period_lc}.nc"
            
            if f_sst.exists():
                key = f"tele_sst_{index}"
                data['obs'][key] = xr.open_dataarray(f_sst)
                data['obs'][f"{key}_pval"] = xr.open_dataarray(f_sst_p)
                data['obs'][f"{key}_fdr"] = calculate_fdr_mask(data['obs'][f"{key}_pval"])

            # SST Index vs PR Field
            f_pr = config.OUTPUT_DIR / f"obs_tele_sst-index-vs-pr-field_{idx_lc}_{period_lc}.nc"
            f_pr_p = config.OUTPUT_DIR / f"obs_tele_sst-index-vs-pr-field_pval_{idx_lc}_{period_lc}.nc"
            
            if f_pr.exists():
                key = f"tele_pr_{index}"
                data['obs'][key] = xr.open_dataarray(f_pr)
                data['obs'][f"{key}_pval"] = xr.open_dataarray(f_pr_p)
                data['obs'][f"{key}_fdr"] = calculate_fdr_mask(data['obs'][f"{key}_pval"])
    except Exception as e:
        print(f"    [!] Warning loading Obs spatial: {e}")

    # 2. Load Models (AMIP & CMIP) using open_mfdataset
    for m_type in ['amip', 'cmip']:
        try:
            # Trends
            trend_files = sorted(config.OUTPUT_DIR.glob(f"{m_type}_*_{'trend'}_{period_lc}.nc"))
            if trend_files:
                data[m_type]['trend'] = xr.open_mfdataset(trend_files, concat_dim="model", combine="nested")['precip_trend']
                if m_type == 'cmip': # Load pvals for CMIP only if needed
                    pval_files = sorted(config.OUTPUT_DIR.glob(f"{m_type}_*_{'trend_pval'}_{period_lc}.nc"))
                    if pval_files:
                        data[m_type]['trend_pval'] = xr.open_mfdataset(pval_files, concat_dim="model", combine="nested")['p_value']

            # Teleconnections
            for index in config.INDICES_TO_PLOT:
                idx_lc = index.lower()
                
                # SST Index vs PR Field
                pr_files = sorted(config.OUTPUT_DIR.glob(f"{m_type}_*_{'tele_sst-index-vs-pr-field'}_{idx_lc}_{period_lc}.nc"))
                if pr_files:
                     data[m_type][f"tele_pr_{index}"] = xr.open_mfdataset(pr_files, concat_dim="model", combine="nested")['correlation']
                     # Add p-values loading here if specifically needed for CSVs
                     pr_p_files = sorted(config.OUTPUT_DIR.glob(f"{m_type}_*_{'tele_sst-index-vs-pr-field_pval'}_{idx_lc}_{period_lc}.nc"))
                     if pr_p_files:
                         data[m_type][f"tele_pr_{index}_pval"] = xr.open_mfdataset(pr_p_files, concat_dim="model", combine="nested")['p_value']

                # PR Index vs SST Field
                sst_files = sorted(config.OUTPUT_DIR.glob(f"{m_type}_*_{'tele_pr-index-vs-sst-field'}_{period_lc}.nc"))
                if sst_files:
                     data[m_type][f"tele_sst_{index}"] = xr.open_mfdataset(sst_files, concat_dim="model", combine="nested")['correlation']
                     
        except Exception as e:
            print(f"    [!] Warning loading {m_type.upper()} spatial: {e}")

    return data



# =============================================================================
# 8. CENTRALIZED DATA LOADING (UNIFIED)
# =============================================================================

def load_ranking_data(period):
    """
    UNIFIED LOADER for 'cleanscript.py' and 'dash_clean.py'.
    Loads:
      1. Individual Model Timeseries (for Ranking/Attribution)
      2. Individual Model Trend Maps (for Categorization: Driest/Wettest)
    """
    print(f"    ... (Lib) Loading Individual Model Data for: {period.upper()} ...")
    cache = {'obs': {'data': None, 'extras': {}, 'trend_map': None}, 'amip': [], 'cmip': []}
    
    # --- Helper to load one file ---
    def _process_file(f, var_name):
        try:
            with xr.open_dataset(f) as ds:
                if var_name in ds:
                    da = ds[var_name].load()
                    da = force_time_to_years(da)
                    extras = {}
                    # Load indices defined in config (for correlation plots)
                    for idx in config.INDICES_TO_PLOT:
                        if idx in ds:
                            extras[idx] = force_time_to_years(ds[idx].load())
                    return da, extras
        except Exception as e:
            print(f"    [!] Error loading {f.name}: {e}")
        return None, None

    # --- 1. Load Observations ---
    obs_ts_file = config.OUTPUT_DIR / f"obs_regression_timeseries_{period.lower()}.nc"
    obs_trend_file = config.OUTPUT_DIR / f"obs_trend_{period.lower()}.nc"
    
    if obs_ts_file.exists():
        da, extras = _process_file(obs_ts_file, "pr_index_study")
        cache['obs']['data'] = da
        cache['obs']['extras'] = extras

    if obs_trend_file.exists():
         with xr.open_dataset(obs_trend_file) as ds:
             if "precip_trend" in ds: cache['obs']['trend_map'] = ds["precip_trend"].load()

    # --- 2. Load Models (Iterative) ---
    
    # A. Pre-load all trend maps into a dictionary for fast lookup
    #    Structure: {'amip': {'filename_stem': DataArray}, ...}
    trend_map_cache = {'amip': {}, 'cmip': {}}
    for m_type in ['amip', 'cmip']:
        pattern = f"{m_type}_*_{'trend'}_{period.lower()}.nc"
        files = list(config.OUTPUT_DIR.glob(pattern))
        for tf in files:
            try:
                ds = xr.open_dataset(tf)
                if 'precip_trend' in ds:
                    trend_map_cache[m_type][tf.stem] = ds['precip_trend'].load()
            except: pass
        # DEBUG: Print how many maps were found
        # print(f"    [Debug] Found {len(files)} trend maps for {m_type}")

    # B. Load Timeseries and Link to Trend Maps
    # Add the current period (e.g., 'march') to ignore list so it isn't picked as the name
    ignore_list = [
        'amip', 'cmip', 'trend', 'mam', 'jja', 'son', 'djf', 
        'pr', 'precip', 'regression', 'timeseries',
        'amon', 'omon', 'day', 'aero', 'historical', 'picontrol',
        'r1i1p1', 'r1i1p1f1', 'gn', 'gr', 
        period.lower()  # <--- CRITICAL ADDITION
    ]
    
    for model_type in ['amip', 'cmip']:
        ts_pattern = f"{model_type}_*_{'regression_timeseries'}_{period.lower()}.nc"
        ts_files = sorted(config.OUTPUT_DIR.glob(ts_pattern))
        
        if not ts_files:
            print(f"    [!] Warning: No timeseries files found for {model_type} ({period})")
            continue

        for f in ts_files:
            if model_type == 'cmip' and "_r2_" in f.name: continue
            
            da, extras = _process_file(f, "pr_index_study")
            if da is not None:
                parts = f.stem.split('_')
                # Extract clean model name (e.g., "Access-CM2")
                name = next((p for p in parts if p.lower() not in ignore_list), "Unknown")
                
                if name == "Unknown":
                    print(f"    [!] Could not extract model name from: {f.stem}")
                    continue

                # --- MATCHING LOGIC ---
                # 1. Try finding the name strictly inside the trend filename parts
                trend_da = None
                for k, v in trend_map_cache[model_type].items():
                    if name in k: 
                        trend_da = v
                        break
                
                # DEBUG: Warn if match failed
                if trend_da is None:
                    print(f"    [Warning] No trend map linked for model: {name}")

                cache[model_type].append({
                    'name': name, 
                    'data': da, 
                    'extras': extras, 
                    'trend_map': trend_da, 
                    'file': f
                })
                
    return cache


def load_ensemble_maps(period):
    """
    UNIFIED LOADER for 'runscript.py'.
    Loads Spatial Maps (Trend, Teleconnections) concatenated for Ensemble analysis.
    """
    print(f"    ... (Lib) Loading Ensemble Spatial Data for: {period.upper()} ...")
    period_lc = period.lower()
    data = {'obs': {}, 'amip': {}, 'cmip': {}}

    # 1. Load Observations
    try:
        # Trend
        f_trend = config.OUTPUT_DIR / f"obs_trend_{period_lc}.nc"
        f_pval = config.OUTPUT_DIR / f"obs_trend_pval_{period_lc}.nc"
        if f_trend.exists(): 
            data['obs']['trend'] = xr.open_dataarray(f_trend)
            data['obs']['trend_pval'] = xr.open_dataarray(f_pval)
            data['obs']['trend_fdr'] = calculate_fdr_mask(data['obs']['trend_pval'])
        
        # Teleconnections (Loop through indices defined in CONFIG)
        for index in config.INDICES_TO_PLOT:
            idx_lc = index.lower()
            
            # PR Index vs SST Field
            f_sst = config.OUTPUT_DIR / f"obs_tele_pr-index-vs-sst-field_{period_lc}.nc"
            f_sst_p = config.OUTPUT_DIR / f"obs_tele_pr-index-vs-sst-field_pval_{period_lc}.nc"
            
            if f_sst.exists():
                key = f"tele_sst_{index}" # Matches runscript key
                data['obs'][key] = xr.open_dataarray(f_sst)
                data['obs'][f"{key}_pval"] = xr.open_dataarray(f_sst_p)
                data['obs'][f"{key}_fdr"] = calculate_fdr_mask(data['obs'][f"{key}_pval"])

            # SST Index vs PR Field
            f_pr = config.OUTPUT_DIR / f"obs_tele_sst-index-vs-pr-field_{idx_lc}_{period_lc}.nc"
            f_pr_p = config.OUTPUT_DIR / f"obs_tele_sst-index-vs-pr-field_pval_{idx_lc}_{period_lc}.nc"
            
            if f_pr.exists():
                key = f"tele_pr_{index}" # Matches runscript key
                data['obs'][key] = xr.open_dataarray(f_pr)
                data['obs'][f"{key}_pval"] = xr.open_dataarray(f_pr_p)
                data['obs'][f"{key}_fdr"] = calculate_fdr_mask(data['obs'][f"{key}_pval"])
                
    except Exception as e:
        print(f"    [!] Warning loading Obs spatial: {e}")

    # 2. Load Models (AMIP & CMIP) - Concatenated
    for m_type in ['amip', 'cmip']:
        try:
            # Trends
            trend_files = sorted(config.OUTPUT_DIR.glob(f"{m_type}_*_{'trend'}_{period_lc}.nc"))
            if trend_files:
                # Combine nested is generally safer for models with different grids if regridded
                
                data[m_type]['trend'] = xr.open_mfdataset(trend_files, concat_dim="model", combine="nested")['precip_trend']
                print(f"    Loading {len(trend_files)} trend files for {m_type.upper()} trend data {data[m_type]['trend']}")

            # Teleconnections
            for index in config.INDICES_TO_PLOT:
                idx_lc = index.lower()
                
                # SST Index vs PR Field
                pr_files = sorted(config.OUTPUT_DIR.glob(f"{m_type}_*_{'tele_sst-index-vs-pr-field'}_{idx_lc}_{period_lc}.nc"))
                if pr_files:
                     data[m_type][f"tele_pr_{index}"] = xr.open_mfdataset(pr_files, concat_dim="model", combine="nested")['correlation']

                # PR Index vs SST Field
                sst_files = sorted(config.OUTPUT_DIR.glob(f"{m_type}_*_{'tele_pr-index-vs-sst-field'}_{period_lc}.nc"))
                if sst_files:
                     data[m_type][f"tele_sst_{index}"] = xr.open_mfdataset(sst_files, concat_dim="model", combine="nested")['correlation']
                     
        except Exception as e:
            print(f"    [!] Warning loading {m_type.upper()} spatial: {e}")

    return data