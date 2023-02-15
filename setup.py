from setuptools import setup

setup(
    name='PatchedMeasCal',
    version='0.2',
    description='Measurement Calibration Patches',
    package_dir={'PatchedMeasCal' : 'src/PatchedMeasCal', 'PatchedMeasCal.benchmarks' : 'src/PatchedMeasCal/benchmarks'},
    author='Alan Robertson',
    packages=['PatchedMeasCal']
)
