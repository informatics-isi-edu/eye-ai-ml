from importlib.metadata import version, PackageNotFoundError
from setuptools_git_versioning import version_from_git
import subprocess
import sys
import os
from pathlib import Path


try:
    __version__ = version("eye_ai")
except PackageNotFoundError:
    # package is not installed
    pass
