import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="advbench",
    version="0.1",
    author="Francesco Croce, Maksym Andriushchenko, Vikash Sehwag",
    author_email="adversarial.benchmark@gmail.com",
    description="This package provides the data for AdvBench together with the model zoo.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AdvBench/advbench",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)