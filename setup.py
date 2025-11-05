from setuptools import setup, find_packages
from pathlib import Path

# Get absolute path to DeepSpeed submodule
base_dir = Path(__file__).parent
deepspeed_path = base_dir / 'DeepSpeed'

setup(
    name='htyllm_pg',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'einops>=0.6.0',
        f'deepspeed @ file://{deepspeed_path.resolve()}',
    ],
)
