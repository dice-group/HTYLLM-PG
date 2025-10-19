from setuptools import setup, find_packages

setup(
    name='htyllm_pg',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'tensorflow==2.19.0',
        'transformers'
    ],
)
