__all__ = [
    "EyeAI",
]

from .eye_ai import EyeAI

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("eye_ai")
except PackageNotFoundError:
    # package is not installed
    pass
