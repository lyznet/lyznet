"""LyZNet public interface."""

from importlib.metadata import PackageNotFoundError, version


try:
    __version__ = version("lyznet")
except PackageNotFoundError:
    __version__ = "uninstalled"


__all__ = ["__version__"]
