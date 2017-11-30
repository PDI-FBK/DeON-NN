"""Setup module for the DeON package."""

from setuptools import setup, find_packages

setup(name='deon',
      version='0.0.1',
      description='Neural net about Definition or not?',
      url='',
      author='',
      author_email='',
      license='Apache License 2.0',
      packages=find_packages(exclude=["tests"]),
      install_requires=[],
      dependency_links=[],
      zip_safe=False,
      tests_require=['pytest'],
      setup_requires=['pytest-runner'])
