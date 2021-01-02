from setuptools import setup

setup(
    name='mnist',
    version='0.0.1',
    packages=['mnist', 'mnist.persistence'],
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
