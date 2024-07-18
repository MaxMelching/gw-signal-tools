import setuptools
import os


# Need read function to paste README into long_description
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


version_config = {
    'version_file': 'gw_signal_tools/_version.py',
    'version_scheme': 'no-guess-dev',
    'local_scheme': 'dirty-tag',
    'fallback_version': 'unknown'  # Or ''?
}


# Great example: https://git.ligo.org/lscsoft/lalsuite/-/blob/master/wheel/setup.py.in
setuptools.setup(
    # ----- Metadata -----
    name='gw_signal_tools',
    use_scm_version=version_config,
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
        # 'Documentation': '',
        'Bug Tracker': 'https://gitlab.aei.uni-hannover.de/fohme/gw-signal-tools/issues'
    },
    packages=setuptools.find_packages(),
    package_data={
        'gw_signal_tools': [
            'PSDs/*.txt',
            'plot_stylesheet.sty'
        ]
    },
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
        'lalsuite',
        'numpy',
        'scipy',
        'matplotlib',
        'gwpy>=3',
        # 'pyseobnr',  # Not entirely sure if needed -> better as extra
        # 'gwsurrogate'  # Not entirely sure if needed -> NOT NEEDED, we only call through lal
        'numdifftools',  # Lightweight, adds no dependencies, thus not included optionally
    ],
    # ----- Optional dependencies -----
    extras_require={
        'dev': [
            'setuptools_scm',  # So that one can test version by
                               # running 'python -m setuptools_scm'
            'mypy',
            'pytest',
            'coverage',
            'pycbc',
        ],
        'pyseobnr': 'pyseobnr',  # Do not install by default, loads for 5s each
                                 # time a Python script is run
        'jupyter': 'jupyter'
    }
)