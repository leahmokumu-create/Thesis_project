# config.py
#
# This is the central configuration file for the unified analysis.
# You do not run this script directly.
# Other scripts (e.g., "run_analysis.py", "make_figures.py") 
# will import these variables.

import xarray as xr
import numpy as np
from pathlib import Path

# =============================================================================
# 1. CORE PATHS
# =============================================================================
# This section defines all the input and output locations for your project.

try:
    # Gets the directory where your script (e.g., run_analysis.py) is located
    SCRIPT_DIR = Path(__file__).parent.resolve()
except NameError:
    # Fallback for running in an interactive environment (like a notebook)
    SCRIPT_DIR = Path.cwd()

# --- Input Data Directories ---
DATA_DIR = SCRIPT_DIR.parent.parent / 'haddata'

# Paths to the specific datasets
SST_PATH = str(DATA_DIR / 'Had' / '*.nc')
GPCC_PATH = str(DATA_DIR / 'gppc' / '*.nc')
AMIP_BASE_DIR = DATA_DIR / 'AMIP'
CMIP_BASE_DIR = Path("/data/reloclim/normal/CMIP6/CMIP")

# --- Output Directory ---
# A single, unified folder for all results and figures
OUTPUT_DIR = SCRIPT_DIR / 'unified_analysis_output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True) # This creates the folder if it doesn't exist
FIGURE_DIR = OUTPUT_DIR / 'figures'
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
# --- Target Grid ---
# A standard 1.0 x 1.0 degree grid for all regridding
DS_TARGET = xr.Dataset(
    {
        "lat": (["lat"], np.arange(-90, 90.1, 1.0)), 
        "lon": (["lon"], np.arange(-180, 180.1, 1.0))
    }
)

# =============================================================================
# 2. ANALYSIS PARAMETERS
# =============================================================================
# These are the main "dials" for your scientific analysis.

START_YEAR = 1981
END_YEAR = 2014
TS_START_YEAR = 1950 # Start year for time series (after 1st diff)
TS_END_YEAR = 2020   # End year for time series
# Periods to analyze (seasonal and monthly)
ANALYSIS_PERIODS = {
    'MAM':   [3, 4, 5],
    'March': [3],
    'April': [4],
    'May':   [5]
}
MONTH_NAMES = {3: 'March', 4: 'April', 5: 'May'} # For plot titles

# --- Statistical Parameters ---
# Use a single, consistent level for all tests (FDR, t-test, etc.)
SIGNIFICANCE_LEVEL = 0.05 
# Model agreement threshold for ensemble "significance"
AGREEMENT_THRESHOLD = 0.75

# =============================================================================
# 3. ATTIRIBUTION MODEL SETTINGS
# =============================================================================

OLS_DRIVER_SETS = {
    'dash': ['IOD', 'WPO', 'Niño1+2'],

}

# =============================================================================
# 3. REGION BOUNDS
# =============================================================================
# A complete dictionary of all named regions used in the analysis.

BOUNDS = {
    # Analysis regions
    'analysis': {'lat_min': -20, 'lat_max': 20, 'lon_min': 25, 'lon_max': 120},
    'study':    {'lat_min': 0,   'lat_max': 12, 'lon_min': 38, 'lon_max': 52},
    'ea':       {'lat_min': -5,  'lat_max': 12, 'lon_min': 30, 'lon_max': 52},
   

    # Index regions (Indian Ocean)
    'wio':      {'lat_min': -10, 'lat_max': 10, 'lon_min': 50, 'lon_max': 70},
    'eio':      {'lat_min': -10, 'lat_max': 0,  'lon_min': 90, 'lon_max': 110},
    #'siod_sw':  {'lat_min': -32, 'lat_max': -24,'lon_min': 55,  'lon_max': 65},
    #'siod_e':   {'lat_min': -20, 'lat_max': -14,'lon_min': 90,  'lon_max': 100},

    # Index regions (Pacific Ocean)
    'wpo':      {'lat_min': -10,  'lat_max': 10,  'lon_min': 130,'lon_max': 150},
    #'indopacific': {'lat_min': -20, 'lat_max': 20, 'lon_min': 40, 'lon_max': 180},
    'nino34':   {'lat_min': -5,  'lat_max': 5,  'lon_min': -170,'lon_max': -120},
    'nino12':   {'lat_min': -10, 'lat_max': 0,  'lon_min': -90, 'lon_max': -80},
    'nino3':    {'lat_min': -5,  'lat_max': 5,  'lon_min': -150,'lon_max': -90},
    'nino4':    {'lat_min': -5,  'lat_max': 5,  'lon_min': 160, 'lon_max': -150},
    #'PDO':      {'lat_min': 20,  'lat_max': 60, 'lon_min': 110, 'lon_max': -100},
   # 'pdo_cool': {'lat_min': 35,  'lat_max': 45, 'lon_min': 160, 'lon_max': -140},
   # 'pdo_warm': {'lat_min': 25,  'lat_max': 45, 'lon_min': -130,'lon_max': -105},
}
# =============================================================================
INDEX_DEFINITIONS = {
    'IOD': ('subtract', 'wio', 'eio', 'standard'),
    'WIO': ('mean', 'wio', 'standard'),
    'EIO': ('mean', 'eio', 'standard'),
    'WPO': ('mean', 'wpo', 'standard'),
   # 'IndoPacific': ('mean', 'indopacific', 'standard'),
    'ENSO': ('mean', 'nino34', 'pacific'),
    'Niño1+2': ('mean', 'nino12', 'pacific'),
    'Niño3': ('mean', 'nino3', 'pacific'),
    'Niño4': ('mean', 'nino4', 'pacific'),
    #'TNI': ('subtract', 'nino12', 'nino4', 'pacific'), 
    #'SIOD': ('subtract', 'siod_sw', 'siod_e', 'standard'),
    #'PDO': ('subtract', 'pdo_warm', 'pdo_cool', 'standard'),
}

# List of indices you actually want to compute and plot
INDICES_TO_PLOT = [
    'IOD', 'WIO', 'EIO', 'WPO', 'ENSO', 
    'Niño1+2', 'Niño3', 'Niño4'
]



# =============================================================================
# 4. PLOTTING CONFIGURATION
# =============================================================================
SMOOTHING_WINDOW = 5
FIG_SIZE_TS = (12, 6)
FIG_SIZE_MAP = (10, 8)