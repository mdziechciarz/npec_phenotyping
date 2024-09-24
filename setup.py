from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of your README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='mark_processor',
    version='0.1.0',
    description='A package for computer vision tasks including instance segmentation, landmark detection, and more.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='neildaniel221270',
    author_email='112754523+neildaniel221270@users.noreply.github.com',
    url='https://github.com/BredaUniversityADSAI/2023-24d-fai2-adsai-group-cv-3',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development :: Libraries',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    keywords='computer vision, instance segmentation, landmark detection, machine learning',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires='>=3.8.13, <3.10.4',
    install_requires=[
        'pandas>=2.0.2,<3.0.0',
        'numpy>=1.24.3,<2.0.0',
        'matplotlib>=3.8.4,<4.0.0',
        'seaborn>=0.1,<0.13.3',
        'scikit-learn>=1.2.2,<2.0.0',
        'tensorflow==2.10',
        'opencv-python-headless>=4.7.0.72,<5.0.0.0',
        'patchify>=0.2.3,<0.3.0',
        'networkx>=2.7.1,<4.0.0',
        'scikit-image>=0.21.0,<0.22.0',
        'skan>=0.1,<0.11.1',
        'typer>=0.12.3,<0.13.0',
        'plotly>=5.14.1,<6.0.0',
        'mlflow>1.20.0,<2.13.1',
        'requests>=2.31.0,<3.0.0',
        'tqdm>=4.65.0,<5.0.0',
        'keras>=2.8.0,<3.0.0',
        'tensorflow-io-gcs-filesystem>=0.28.0,<0.37.0',
        'torch>=1.0.0',
        'torchvision>=0.15.1,<0.16.0',
        'pre-commit>=3.2.0,<4.0.0',
        'geopandas>=0.13.0,<0.14.0',
        'azureml-core>=1.50.0,<1.57.0',
        'azure-identity>=1.12.0,<2.0.0',
        'azure-ai-ml>=1.4.0,<2.0.0',
        'poetry2conda>=0.3.0,<0.4.0',
        'azureml-mlflow==1.56.0',
        'opencensus-ext-azure==1.1.13',
        'azureml-inference-server-http>=0.1.5',
    ],
    extras_require={
        'dev': ['pytest>=8.2.0'],
    },
    package_data={
        'compv': [
            'data/*',
            'models/*',
            'notebooks/*',
            'docs/*'
        ],
    },
    entry_points={
        'console_scripts': [
            'mark-processor=src.main:main',
        ],
    },
)
