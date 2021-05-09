from setuptools import find_packages, setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="ml_project",
    packages=find_packages(),
    version="0.1.0",
    description="Vladimir Bulaev. Homework01",
    author="bulaevvi",
    install_reqs=required,
    license="MIT",
)