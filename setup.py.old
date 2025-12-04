from setuptools import setup, find_packages

setup(
    name="depman",
    version="0.1.0",
    description="CLI extension for Gitman: Wraps commands and checks for dependency updates",
    author="Your Name",
    packages=find_packages(),  # Finds depman/ as package
    install_requires=[
        "click>=8.0",
        "gitman>=0.11.0",
        "gitpython>=3.1.0",
    ],
    entry_points={
        "console_scripts": [
            "depman=depman:cli",  # Points to depman/__init__.py
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)