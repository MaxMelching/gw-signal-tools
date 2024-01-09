import setuptools

version = '0.0.1'



setuptools.setup(
    name='gw_signal_tools',
    version=version,
    author='Frank Ohme, Max Melching',
    # author_email='',
    description='Tools for GW Data Analysis',
    # long_description='',
    url='https://gitlab.aei.uni-hannover.de/fohme/gw-signal-tools',
    packages=setuptools.find_packages(),
    include_package_data=True,
    zip_safe=False,
    # python_requires='>=3.10',
    python_requires='==3.11.5',  # Installation is verified to work with that
    # python_requires='<3.11',
    install_requires=[
        # 'lalsuite',
        'lalsuite[lalinference]',
        'numpy',
        'scipy',
        'matplotlib'
    ],
    extras_require={
        'jupyter': 'jupyter',
        'pyseobnr': 'pyseobnr',
        'dev': [
            'pycbc',
            'mypy',
            # 'pytest'
        ]
    }  # Install each of these options via pip install -e .[option]
)