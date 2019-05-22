from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name="dataprep",
    version="0.1",
    author="Weiyuan Wu",
    author_email="youngw@sfu.ca",
    description="Dataprep",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sfu-db/dataprep",
    packages=find_packages(),
    classifiers=[
        "Environment :: Console",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Cython",
        "Topic :: Scientific/Engineering"
    ],
)
