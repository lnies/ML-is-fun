#
#   Definition of global variables
#   for generating the sets, etc...
#

from . import *

### Variables for creating the training, dev, and test sets
# Distribution parameters
GAMMA1, SIGMA1, L1 = 0.345, 0.07, 0.18 # parameters of the first voigt profile
GAMMA2, SIGMA2, L2 = 0.36, 0.08, 0.30 # parameters of the second voigt profile
MEAN_ENERGY = 2014.00 # in (eV) is mean energy of K_alpha1
MEAN_ENERGY_SHIFT = 0.85 # in (eV) is mean energy shift between K_alpha1 and K_alpha2
# Parameters determining size of library and # of spectra
MAXNUMSTATES = 3
## Definition of grids involved
# Grid for plotting the spectra
ENERGY_BIN_WIDTH_PLOT = 0.01 # in (eV)
MIN_ENERGY_PLOT = 2008.00 # in (eV) left boundary for plotting the spectra
MAX_ENERGY_PLOT = 2018.00 # in (eV) right boundary for plotting the spectra
GRID_PLOT = np.arange(MIN_ENERGY_PLOT, MAX_ENERGY_PLOT, ENERGY_BIN_WIDTH_PLOT) # Creation of grid for plotting
# Grid for creating the features
ENERGY_BIN_WIDTH_FEATURE = 0.05 # in (eV) is grid size of discretizing the spectrum (feature grid)
MIN_ENERGY_FEATURE = 2008.00 # in (eV) minimal value for the feature grid
MAX_ENERGY_FEATURE = 2018.00 # in (eV) maximal value for the feature grid
GRID_FEATURES = np.arange(MIN_ENERGY_FEATURE, MAX_ENERGY_FEATURE, ENERGY_BIN_WIDTH_FEATURE)
# Grid for creating the target label
ENERGY_BIN_WIDTH_TARGET = 0.01 # in (eV) is grid size of discretizing the target grid
MIN_ENERGY_LABEL = 2013.00 # in (eV) is minimal energy value for target grid
MAX_ENERGY_LABEL = 2016.00 # in (eV) is maximal energy value for target grid 
GRID_TARGET = np.arange(MIN_ENERGY_LABEL, MAX_ENERGY_LABEL, ENERGY_BIN_WIDTH_TARGET)
## Parameters for splitting the library
TRAIN_SPLIT = 0.8 # splitting ratio of the library in favor of training set NOT IMPLEMENTED YET
DEV_SPLIT = 0.1 # splitting ratio of the library in favor of dev set NOT IMPLEMENTED YET
TEST_SPLIT = 0.1 # splitting ratio of the library in favor of test set NOT IMPLEMENTED YET
## Parameters for applying the Poisson noise
POISSONCNTS = 1 #0.5*1e3 
## Other global variables
legend = ['states', 'energies', 'energies_binned', 'splittings', 'ratios', 'features', 'energy_label', 'plot']

example = 'banana'