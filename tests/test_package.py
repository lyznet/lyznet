from importlib.metadata import version

import lyznet


def test_package_reports_installed_version():
    assert lyznet.__version__ == version("lyznet")
