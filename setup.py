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
        # 'test': 'pytest'
    }  # Hope this is how it works -> seems to be given as option via pip install -test -> nope, vai pip install -e .[option]
)