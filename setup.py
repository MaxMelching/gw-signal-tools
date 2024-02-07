import setuptools
import subprocess
import os


# Automatically get version and make it available in package
version = (
     subprocess.run(['git', 'tag', '--points-at', 'HEAD'], stdout=subprocess.PIPE)
     .stdout.decode('utf-8')
     .strip()
)

if version == '':
    version = (
        subprocess.run(['git', 'describe', '--tags', '--abbrev=0'], stdout=subprocess.PIPE)
        .stdout.decode('utf-8')
        .strip()
    )

if version[0] == 'v':
    version = version[1:]

# Write to file that is used in __init__.py
with open('gw_signal_tools/_version.py', mode='wt') as f:
    f.write(f'__version__ = \'{version}\'')
    f.close()


# We paste README into long_description, need read function for this
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


# TODO: rather use pyproject.toml instead of setup.py?

setuptools.setup(
    name='gw_signal_tools',
    version=version,
    author='Frank Ohme, Max Melching',
    author_email='max.melching@aei.mpg.de',
    description='Tools for GW Data Analysis',
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    url='https://gitlab.aei.uni-hannover.de/fohme/gw-signal-tools',
    packages=setuptools.find_packages(),
    package_data={'gw_signal_tools': ['PSDs/*.txt']},
    include_package_data=True,
    zip_safe=False,
    python_requires='==3.11.5',  # Installation is verified to work with that
    install_requires=[
        'lalsuite[lalinference]',
        'numpy',
        'scipy',
        'matplotlib'  # TODO: think about moving this to dev
    ],
    extras_require={
        'dev': [
            'pycbc',
            'mypy',
            'pytest',
            'coverage'
        ],
        'jupyter': 'jupyter',
        # 'pyseobnr': 'pyseobnr',  # Apparently causes error
    }
)