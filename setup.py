from setuptools import setup

setup(
    name="industrial_robotics_simulator",
    version="1.0.0",
    author="Matthias De Ryck",
    author_email="matthias.deryck@kuleuven.be",
    description="A package for simple robot control as an addition to the lecture Industrial Robotics at the KU Leuven Campus in Bruges",
    url="https://github.com/MatthiasDR96/industrial_robotics_simulator.git",
    packages=['industrial_robotics_simulator', 'industrial_robotics_simulator.arms', 'industrial_robotics_simulator.controllers'],
    package_dir={'':  'src'}
)
