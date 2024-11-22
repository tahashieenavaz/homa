from setuptools import setup
from setuptools import find_packages
from sys import argv

with open("README.md") as fh:
    description = fh.read()

with open("version.txt", "r") as fh:
    current_version = float(fh.readline())

with open("version.txt", "w") as fh:
    next_version = round(current_version + 0.01, 2)
    fh.write(str(next_version))

setup(
    name="homa",
    maintainer="Taha Shieenavaz",
    maintainer_email="tahashieenavaz@gmail.com",
    version=next_version,
    packages=find_packages(),
    install_requires=["torchvision", "torch"],
    long_description=description,
    long_description_content_type="text/markdown",
)
