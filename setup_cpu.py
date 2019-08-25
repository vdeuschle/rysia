import sys
from pathlib import Path
from setuptools import setup, find_namespace_packages


if sys.version_info.major is not 3 or sys.version_info.minor is not 6:
    raise Exception('rysia requires Python version 3.6')

ROOT = Path(__file__).parent
SRC = ROOT / 'src'

requirements_path = ROOT / 'requirements-cpu.txt'

with requirements_path.open() as f:
    requirements = [line.rstrip() for line in f]

setup(name='rysia',
      description='a declarative benchmarking framework for deep learning systems',
      url='https://github.com/vdeuschle/rysia',
      author='Vincent Deuschle',
      version='0.1',
      license='Apache License 2.0',
      python_requires="== 3.6",
      package_dir={'': 'src'},
      packages=find_namespace_packages(where=SRC, include=['rysia*']),
      install_requires=requirements,
      entry_points={
          'console_scripts': ['rysia=rysia.main:main'],
        }
      )
