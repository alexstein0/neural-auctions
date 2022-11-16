from setuptools import setup
from os import path
import sys

from io import open

here = path.abspath(path.dirname(__file__))
sys.path.insert(0, path.join(here, "almost-unique-id"))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(name="almost_unique_id",
      version="v0.0.1",
      description="Almost Unique IDs",
      author="Anon",
      keywords=["run ids"],
      long_description=long_description,
      long_description_content_type="text/markdown",
      py_modules=["almost_unique_id"],
      python_requires=">=3.7",
      install_requires=[],
      license="MIT")
