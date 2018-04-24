from distutils.core import setup

setup(
    name='MLforXES',
    version='0.1',
    packages=['MLforXES'],
    author='Lukas',
    requires=['astropy', 'plotly', 'sklearn', 'tarfile', 'dill', 'msgpack', 'h5py', 'lmfit', 'peakutils']
)