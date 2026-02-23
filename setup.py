from setuptools import setup, find_packages
from version import __version__ as VERSION

DESCRIPTION = "Runtime Uncertainty estimation for AI models"
LONG_DESCRIPTION = "Uncertainty estimation methods for Deep Neural Networks (DNNs) in computer vision and natural language processing, with a focus on out-of-distribution (OOD) detection."

# Setting up
setup(
    name="runia_core",
    version=VERSION,
    author="Daniel Montoya",
    author_email="<daniel-alfonso.montoyavasquez@cea.fr>",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    python_requires=">=3.9",
    packages=find_packages(),
    keywords=[
        "Uncertainty",
        "Artificial Intelligence",
        "Confidence",
        "Trustworthiness",
        "Out-of-distribution detection",
        "DNNs",
    ],
    classifiers=[
        "Development Status :: 4 - Alpha",
        "Intended Audience :: Research and industrial community",
        "Programming Language :: Python :: 3",
        "Operating System :: Linux",
    ],
)
