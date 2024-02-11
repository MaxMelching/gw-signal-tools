import setuptools
import os


# Need read function to paste README into long_description
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


# Great example: https://git.ligo.org/lscsoft/lalsuite/-/blob/master/wheel/setup.py.in
setuptools.setup(
    # ----- Metadata -----
    name='gw_signal_tools',
    use_scm_version=True,
    author='Frank Ohme, Max Melching',
    author_email='max.melching@aei.mpg.de',
    maintainer='Max Melching',
    maintainer_email='max.melching@aei.mpg.de',
    description='Tools for GW Data Analysis',
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    url='https://gitlab.aei.uni-hannover.de/fohme/gw-signal-tools',
    project_urls={
        'Source Code': 'https://gitlab.aei.uni-hannover.de/fohme/gw-signal-tools',
        # "Documentation": '',
        "Bug Tracker": 'https://gitlab.aei.uni-hannover.de/fohme/gw-signal-tools/issues'
    },
    packages=setuptools.find_packages(),
    package_data={'gw_signal_tools': ['PSDs/*.txt']},
    include_package_data=True,
    # ----- Dependencies for installation -----
    setup_requires=[
        'setuptools>=64',
        'setuptools_scm>=8',
        'wheel'
    ],
    # ----- Dependencies for package -----
    python_requires='==3.11.5',  # Installation is verified to work with that
    install_requires=[
        'lalsuite[lalinference]',
        'numpy',
        'scipy',
        'matplotlib'
    ],
    # ----- Optional dependencies -----
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