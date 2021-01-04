from setuptools import setup, find_packages

setup(
    name='mnist',
    version='1.0.0',
    packages=find_packages(),
    url='',
    license='',
    author='André Claudino',
    author_email='',
    description='',
    install_requires=[
        "tensorflow-gpu==2.4.0",
        "click==7.1.2"
    ],
    entry_points={
        'console_scripts': [
            'train=mnist.training.main:main',
            'validate=mnist.validating.main:main'
        ]
    }
)
