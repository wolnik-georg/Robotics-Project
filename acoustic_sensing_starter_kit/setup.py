"""
Setup script for Acoustic Sensing package
"""

from setuptools import setup, find_packages

with open("Readme.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip() for line in fh if line.strip() and not line.startswith("#")
    ]

setup(
    name="acoustic-sensing",
    version="1.0.0",
    author="Georg Wolnik",
    author_email="georg.wolnik@example.com",
    description="Advanced acoustic sensing system for geometric reconstruction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wolnik-georg/Robotics-Project",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
        ],
    },
    entry_points={
        "console_scripts": [
            "acoustic-sense=acoustic_sensing.legacy.C_sense:main",
            "acoustic-record=acoustic_sensing.legacy.A_record:main",
            "acoustic-train=acoustic_sensing.legacy.B_train:main",
        ],
    },
)
