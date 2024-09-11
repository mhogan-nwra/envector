"""Run tests using Nox

Quick Usage
-----------

```
pdm run nox -s <SESSION_NAME>-<VERSION>
```

Description
-----------

You have options on how to use Nox depending on which applications are on your platform. If you have all supported
Python versions installed and in your $PATH, then you can use the `noxfile.test` method like so

```
pdm run nox -s tests_venv       # Or
pdm run nox -s tests_virtualenv
```

If you have a limited installation set, you can install the Anaconda, Mamba, or micromamba and run

```
pdm run nox -s tests_conda        # Or
pdm run nox -s tests_mamba        # Or
pdm run nox -s tests_micromamba
```

"""
from pathlib import Path

import nox
import os

os.environ.update({"PDM_IGNORE_SAVED_PYTHON": "1"})
supported_pythons = ["3.9", "3.10", "3.11", "3.12"]


@nox.session(
    python=supported_pythons,
    venv_backend="venv"
)
def tests_venv(session: nox.Session) -> None:
    """Test environment the venv venv_backend"""
    _pdm_install_test_group(session)
    _unifed_test(session)


@nox.session(
    python=supported_pythons,
    venv_backend="virtualenv"
)
def tests_virtualenv(session: nox.Session) -> None:
    """Test environment the virtualenv venv_backend"""
    _pdm_install_test_group(session)
    _unifed_test(session)


@nox.session(
    python=supported_pythons,
    venv_backend="conda",
)
def tests_conda(session: nox.Session) -> None:
    """Test environment with the conda venv_backend"""
    _tests_conda_like(session)


@nox.session(
    python=supported_pythons,
    venv_backend="micromamba",
)
def tests_micromamba(session: nox.Session) -> None:
    """Test environment with the micromamba venv_backend"""
    _tests_conda_like(session)


@nox.session(
    python=supported_pythons,
    venv_backend="mamba",
)
def tests_mamba(session: nox.Session) -> None:
    """Test environment with the mamba venv_backend"""
    _tests_conda_like(session)


def _pdm_install_test_group(session: nox.Session) -> None:
    session.run_always("pdm", "install", "-dG", "test", external=True)


def _tests_conda_like(session: nox.Session) -> None:
    session.run_always("pdm", "build", external=True)
    session.install("--upgrade", "pip")
    session.run_always("pip", "install", "-f", os.fspath(Path.cwd() / "dist"), "envector")
    session.install("pytest", "pytest-cov", "pytest-pep8", "hypothesis")
    _unifed_test(session)

def _unifed_test(session: nox.Session) -> None:
    """The unified command to run the test suite"""
    session.run("pytest", "-rxsXf", "--doctest-modules", "src", "tests")
