from setuptools import setup, find_packages

setup(
    name="minigraphs",   
    version="0.1",
    packages=find_packages(),
    include_package_data=True,  
    install_requires=[          
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "scikit-learn",
        "networkx",
        "click",
        "pydantic<=2.8.0"
    ],
)
