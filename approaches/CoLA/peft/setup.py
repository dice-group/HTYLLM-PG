from __future__ import annotations

from distutils.sysconfig import get_python_lib
from pathlib import Path

from setuptools import find_packages, setup
from setuptools.command.develop import develop as _develop


SOURCE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SOURCE_ROOT.parent
PTH_FILENAME = "peft_local.pth"

_subpackages = find_packages(include=["tuners", "tuners.*", "utils", "utils.*"])
packages = ["peft", *[f"peft.{pkg}" for pkg in _subpackages]]


def _write_local_pth() -> None:
    target = Path(get_python_lib()) / PTH_FILENAME
    target.write_text(f"{PROJECT_ROOT}\n")


class develop(_develop):
    def install_for_development(self) -> None:  # type: ignore[override]
        super().install_for_development()
        _write_local_pth()


setup(
    name="peft",
    version="0.12.0",
    packages=packages,
    package_dir={"peft": "."},
    include_package_data=True,
    cmdclass={"develop": develop},
)
