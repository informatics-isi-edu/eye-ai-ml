from importlib.metadata import version, PackageNotFoundError
import subprocess
import sys
import os
from pathlib import Path


try:
    __version__ = version("eye_ai")
except PackageNotFoundError:
    # package is not installed
    pass
