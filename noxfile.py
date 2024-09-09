"""Run tests using Nox"""
import nox
import os

os.environ.update({"PDM_IGNORE_SAVED_PYTHON": "1"})

@nox.session(
    python=["3.9", "3.10", "3.11", "3.12"],
)
def tests(session: nox.Session) -> None:
    session.run_always("pdm", "install", "-G", "test", external=True)
    session.run("pytest", "-rxsXf", "--doctest-modules", "src", "tests")
