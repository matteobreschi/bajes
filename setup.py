from setuptools import setup, find_packages
import logging
import shutil
import glob
import re
import sys
import os

__ascii_art__ = """
                    ....
              ....--::::-==-:.        .::..
           ..---::.         :=:..    .:  :..
        ...--.     ..... ..   :-::   .:   .:.
       :::-.   .---:::::-::---  :=:.   :    :.
     .-=.   .---..          .:+   =.. ..:    :.
   .:=:    --:.               :=  .=..  .:    :.
  :-=    -=:.                 .:=  + :  .:    :.
 :.=    --.   .:..:.           :+ --:    .:   :.
..+. .--:.  .:.   .:  .        :.-:.     .:   ...
 .:--:..  .::   .:.            ...       .:   ...
         . :  .:.                        .:   : .
        ..: .. .                        .:    :.
        ... ...                        .:    :..
        .:   :.         .             .:   :
        .:    :.        .....:.. ..  ..   ...
         .:    :..      ...    .. .........
          .:    :.        ............
           .:.    .:.
             .::     ..::.   ...
               .:..       .... ..:.
                  ......          .-.
                        .:::.....::..
"""

# set logger
logging.basicConfig(level=logging.INFO, format='%(message)s')

# check python >= 3.7
py_version = sys.version_info
if py_version < (3, 7):
    sys.exit("Python < 3.7 is not supported, aborting setup")

# get directory path
dir_path = os.path.dirname(os.path.realpath(__file__))

# get version from __init__
inis    = open(os.path.join(dir_path, 'bajes', '__init__.py')).read()
VERS    = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo      = re.search(VERS, inis, re.M)
VERSION = mo.group(1)

# Tidy up the project root if installing
if 'install' in sys.argv:

      logging.info("Running Bajes {} installation with Python {}.{}".format(VERSION, py_version.major, py_version.minor))
      logging.info("cleaning project root")
      CLEAN_FILES = './build ./dist ./*.pyc ./*.tgz ./*.egg-info'.split(' ')

      for path_spec in CLEAN_FILES:
          # Make paths absolute and relative to this path
          abs_paths = glob.glob(os.path.normpath(os.path.join(dir_path, path_spec)))
          for path in [str(p) for p in abs_paths]:
              if not path.startswith(dir_path):
                  # Die if path in CLEAN_FILES is absolute + outside this directory
                  raise ValueError("{} is not a path inside {}".format(path, dir_path))
              logging.info("removing {}".format(os.path.relpath(path)))
              shutil.rmtree(path)

setup(# metadata
      name='bajes',
      version=VERSION,
      description='Bayesian Jenaer Software',
      long_description=open(os.path.join(dir_path, 'README.md')).read(),
      long_description_content_type="text/x-rst",
      author='Matteo Breschi',
      author_email='matteo.breschi@uni-jena.de',
      url='https://github.com/matteobreschi/bajes',
      license='MIT',

      # list of packages and data
      packages=find_packages(),
      package_dir={"bajes": "bajes"},
      package_data={"bajes": ["pipe/data/gw/asd/events/*/*.txt",
                              "pipe/data/gw/asd/design/*.txt",
                              "pipe/data/gw/spcal/events/*/*.txt",
                              "pipe/data/kn/filter/AT2017gfo/*.txt",
                              "obs/kn/fluxfactors/*.dat"]},

      # make scripts executable
      scripts   = ['bajes/pipe/scripts/bajes_core.py',
                   'bajes/pipe/scripts/bajes_parallel_core.py',
                   'bajes/pipe/scripts/bajes_pipe.py',
                   'bajes/pipe/scripts/bajes_inject.py',
                   'bajes/pipe/scripts/bajes_read_gwosc.py',
                   'bajes/pipe/scripts/bajes_postproc.py'],

      # set mandatory requirements
      python_requires='>=3.7',
      install_requires=['numpy>=1.18.0',
                        'scipy>=1.4.0',
                        'astropy>=4.0.0']
      )

if 'install' in sys.argv:
      logging.info("Bajes succesfully installed.")
      logging.info(__ascii_art__)
