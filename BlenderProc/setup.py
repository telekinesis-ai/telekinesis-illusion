from setuptools import setup, find_packages
import os

# Extract version from blenderproc/version.py
here = os.path.abspath(os.path.dirname(__file__))
version = {}
with open(os.path.join(here, "blenderproc", "version.py")) as fp:
    exec(fp.read(), version)

with open(os.path.join(here, "README.md")) as fp:
    long_description = fp.read()

setup(
    name="blenderproc",
    version=version["__version__"],
    url="https://github.com/DLR-RM/BlenderProc",
    author="Maximilian Denninger, Dominik Winkelbauer, Martin Sundermeyer",
    maintainer="Dominik Winkelbauer",
    packages=find_packages(
        exclude=[
            "docs",
            "examples",
            "external",
            "images",
            "resources",
            "scripts",
            "tests",
        ]
    ),
    include_package_data=True,
    entry_points={
        "console_scripts": ["blenderproc=blenderproc.command_line:cli"],
    },
    install_requires=[
        "setuptools==80.9.0",
        "pyyaml==6.0.1",
        "requests==2.32.5",
        "matplotlib==3.9.0",
        "numpy==2.4.2",
        "Pillow>=10.3.0",
        "h5py==3.11.0",
        "progressbar==2.5",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
)
