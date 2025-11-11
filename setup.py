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
        'torch>=2.9.0',
        'einops>=0.6.0',
        f'deepspeed @ file://{deepspeed_path.resolve()}',
        'matplotlib>=3.10.7',
        'tqdm>=4.67.1',
        'tokenizers>=0.22.1',
        # 'lm_eval==0.4.8',     #uncomment if deepspeed reqs don't cover this, ATTENTION: deepspeed reqs mention 0.3.0, if problems arise
        'transformers>=4.57.1',
        'numpy>=2.3.4'
    ],
)
