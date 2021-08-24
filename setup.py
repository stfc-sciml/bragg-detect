from setuptools import setup, find_packages

# find version
with open('bragg_detect/__init__.py') as f:
    version = f.read().splitlines()[-1].split("'")[-2]

# framework dependencies
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# setup
setup(
    name='bragg-detect',
    version=version,
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    license='BSD-3-Clause License',
    author='Kuangdai Leng, Pascal Manuel, Jeyan Thiyagalingam',
    author_email='kuangdai.leng@stfc.ac.uk',
    description='Detecting Bragg peaks in 3D X-ray/neutron images'
)
