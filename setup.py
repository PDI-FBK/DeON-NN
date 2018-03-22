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
      install_requires=[
            'click==6.7',
            'tensorflow-gpu==1.4.1'
      ],
      dependency_links=[],
      zip_safe=False,
      test_suite='test')
