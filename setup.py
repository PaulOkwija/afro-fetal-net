from setuptools import find_packages, setup

setup(
    name="fetal_ai",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
)
