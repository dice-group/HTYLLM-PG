from setuptools import setup

setup(
    name="peft",
    version="0.12.0",
    py_modules=[
        "auto",
        "config",
        "helpers",
        "import_utils",
        "mapping",
        "mixed_model",
        "peft_model",
    ],
    packages=["tuners", "utils"],
)
