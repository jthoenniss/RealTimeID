from setuptools import setup, find_packages

setup(
    name="RealTimeID",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.3",
        "scipy>=1.10.",
        "maplotlib>=3.3.4"
    ],
)
