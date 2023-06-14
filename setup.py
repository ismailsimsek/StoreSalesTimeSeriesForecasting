import os

from setuptools import setup, find_packages

setup_py_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(setup_py_dir)

with open("requirements.txt", "r") as fh:
    requirements = fh.readlines()
    install_requires = [req.strip() for req in requirements if
                        not req.strip().startswith('#') and not req.strip().startswith('http') and req.strip()]
    dependency_links = [req.strip() for req in requirements if req.strip().startswith('http')]

setup(name='mymllib',
      version="1.0.0",
      description="A library for generic functions and utils for ML!",
      packages=find_packages(),
      install_requires=requirements,
      test_suite='tests',
      scripts=[],
      zip_safe=False,
      license='',
      author='AI Team',
      author_email='machile@learning.com',
      include_package_data=True,
      long_description="A python module for ML tasks.",
      dependency_links=dependency_links
      )
