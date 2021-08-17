import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="robustbench",
    version="0.1",
    author="Francesco Croce, Maksym Andriushchenko, Vikash Sehwag",
    author_email="adversarial.benchmark@gmail.com",
    description="This package provides the data for RobustBench together with the model zoo.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RobustBench/robustbench",
    packages=setuptools.find_packages(),
    install_requires=[
        'requests~=2.25.0',
        'numpy~=1.19.4',
        'Jinja2~=2.11.2',
        'tqdm~=4.56.1',
        'pandas~=1.1.0',
        'autoattack @ git+https://github.com/fra31/auto-attack.git@f551404cf7ed1d6e17d3041d5709ad23f6dbf352#egg=autoattack'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
