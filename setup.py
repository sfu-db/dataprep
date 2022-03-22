from skbuild import setup

def build():
    setup(
        name="dataprep",
        version="0.4.2",
        url="https://github.com/sfu-db/dataprep",
        author="SFU Database System Lab <dsl.cs.sfu@gmail.com>",
        install_requires=["Levenshtein >= 0.16.0, < 0.18.1"],
        author_email="dsl.cs.sfu@gmail.com",
        description="DataPrep lets you prepare your data using a single library with a few lines of code.",

        license="GPL",
        license_file = "COPYING",
        classifiers=[
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)"
        ],

        packages=["dataprep"],
        package_dir={'':''},
        zip_safe=True,
        include_package_data=True,
        python_requires=">=3.7"
    )

if __name__ == '__main__':
    build()
