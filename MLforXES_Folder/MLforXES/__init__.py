"""
Import functions and tools needed
"""
import os
import plotly.offline as py
import plotly.tools as tls
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import pandas as pd
import matplotlib
import math
import random
import scipy.stats as sc
from scipy import interpolate
from scipy.special import wofz
from astropy.modeling.functional_models import Voigt1D 
from scipy.interpolate import UnivariateSpline
# several sklearn functions
from sklearn.neural_network import MLPRegressor as MLPR
from sklearn.neural_network import MLPClassifier as MLPC
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.datasets import make_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
import sklearn as skl
import time
from sklearn.externals import joblib
# Redirect stdout
import io 
import sys
# Use subpreocesses
import subprocess
# Use tgz to zip large files into compressed containers
import tarfile
# Use garbage collect to clear unused variables and free memory
import gc
# Export data as different file formats
import msgpack
import h5py
import dill # Saves everything! 
# Extra functions for dictionaries
from itertools import chain
from collections import defaultdict
# Control figure size
matplotlib.rcParams['figure.figsize']=(7,5)
py.init_notebook_mode()
# Use plotly as gifure output
def plotly_show():
    fig = plt.gcf()
    plotlyfig = tls.mpl_to_plotly(fig,resize=True)
    plotlyfig['layout']['showlegend']=True
    py.iplot(plotlyfig)