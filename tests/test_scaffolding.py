import importlib

import pytest


@pytest.mark.parametrize("package", ["magma", "bench", "baselines"])
def test_package_imports(package: str) -> None:
    assert importlib.import_module(package) is not None


def test_version_is_set() -> None:
    import magma

    assert magma.__version__
