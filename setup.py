from setuptools import setup, find_packages

setup(
    name="ooicgsn_glider_dvl",
    version="0.0.1a",
    author="Andrew Reed",
    authoer_email="areed@whoi.edu",
    description="A package for processing OOI CGSN Glider ADCP (DVL) data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPLv3",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    python_requires=">=3.9.7"
)