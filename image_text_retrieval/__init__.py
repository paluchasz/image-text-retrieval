import importlib.metadata

__version__ = importlib.metadata.version(__name__)

PACKAGE_NAME = __name__

__all__ = ["__version__", "PACKAGE_NAME"]
