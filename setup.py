import setuptools

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name="robustbench",
    version="1.0",
    author=
    "Francesco Croce, Maksym Andriushchenko, Vikash Sehwag, Edoardo Debenedetti",
    author_email="adversarial.benchmark@gmail.com",
    description=
    "This package provides the data for RobustBench together with the model zoo.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RobustBench/robustbench",
    packages=setuptools.find_packages(),
    install_requires=[
        'torch>=1.7.1', 'torchvision>=0.8.2', 'torchdiffeq', 'geotorch', 'requests~=2.25.0',
        'numpy>=1.19.4', 'Jinja2~=2.11.2', 'tqdm~=4.56.1', 'pandas~=1.2.0',
        'autoattack @ git+https://github.com/fra31/auto-attack.git@6482e4d6fbeeb51ae9585c41b16d50d14576aadc#egg=autoattack'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
)
