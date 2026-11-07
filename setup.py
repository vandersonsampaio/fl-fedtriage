from setuptools import find_packages, setup

with open("requiriments.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="fedtriage",
    version="0.1.0",
    description="TRIAGE — Threshold-based Robust Inclusion via Agreement of Gradient Evidence (FL robust aggregation)",
    packages=find_packages(include=["src", "src.*"]),
    install_requires=requirements,
    python_requires=">=3.10",
)
