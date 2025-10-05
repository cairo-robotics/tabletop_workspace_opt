from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup
from setuptools import find_packages
d = generate_distutils_setup(
    name='shared_autonomy_workspace_optimization',
    packages=["find_packages('src')"],
    package_dir={'': 'src'},
    install_requires=[],
)
setup(**d)
