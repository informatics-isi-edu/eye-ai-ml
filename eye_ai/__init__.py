from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("eye-ai")
except PackageNotFoundError:
    # package is not installed
    pass

print(f"Init version{__version__}")