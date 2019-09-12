from setuptools import setup, find_packages
from pipenv.project import Project
from pipenv.utils import convert_deps_to_pip

pfile = Project(chdir=False).parsed_pipfile
requirements = convert_deps_to_pip(pfile['packages'], r=False)
test_requirements = convert_deps_to_pip(pfile['dev-packages'], r=False)

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
    install_requires=requirements,
    tests_requires=test_requirements,
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
