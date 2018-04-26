from setuptools import setup

setup(
    name='gym_mosquitoes',
    version='0.0.1',
    packages=['gym_mosquitoes', 'gym_mosquitoes.envs'],
    install_requires=[
        'gym',
        'numpy',
        'matplotlib'
    ]
)
