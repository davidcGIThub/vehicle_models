from setuptools import setup, find_packages

setup(
    name='vehicle_simulator',
    version='0.1.0',
    packages=find_packages(include=['vehicle_simulator', 'vehicle_simulator.*'])
)