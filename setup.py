from setuptools import setup, find_packages

VERSION = '0.1'

setup(
    name='caesar',
    version=VERSION,
    description='Caesar does cool stuff.',
    url='https://bitbucket.org/rthompson/caesar',
    author='Robert Thompson',
    author_email='rthompsonj@gmail.com',
    license='not sure',

    classifiers=[],
    install_requires=[],
    keywords='',
    entry_points={
        'console_scripts':['caesar = caesar.command_line:run']
    },
    packages=find_packages(),
)

