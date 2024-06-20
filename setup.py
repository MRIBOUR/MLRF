from setuptools import setup, find_packages

setup(
    name='mlrf',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scikit-learn',
        'scikit-image',
        'opencv-python',
        'matplotlib'
    ],
)