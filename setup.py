import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="robustbench",
    version="0.2.1",
    author="Francesco Croce, Maksym Andriushchenko, Vikash Sehwag",
    author_email="adversarial.benchmark@gmail.com",
    description="This package provides the data for RobustBench together with the model zoo.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RobustBench/robustbench",
    packages=setuptools.find_packages(),
    install_requires=[
        'requests~=2.23.0',
        'numpy~=1.20.3',
        'Jinja2~=2.11.2',
        'tqdm~=4.56.1',
        'pandas~=1.1.0',
        'autoattack @ git+https://github.com/fra31/auto-attack.git@9b264b52bb65c373373727f532865a5551ab9c02#egg=autoattack',
        "tensorboard~=2.5.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
