import setuptools
import os


# We paste README into long_description, need read function for this
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setuptools.setup(
    name='gw_signal_tools',
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
        # 'setuptools_scm',  # Is already covered under build-system in toml, thus not needed here
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

# This is strange: if I try to move stuff entirely to pyprojet.toml, an error occurs.
# However, if I comment build-system stuff in toml, versioning does not work...
# -> ahhh, figured it out: problem occurs with empty, but existing setup in
# combination with toml. In that case, two setup files exist (formally), which
# leads to a conflict. Renaming this file here and using just toml works fine
# -> more detailed: [project] section of toml is problem; this is also where
# dynamic part lies, so not sure if this has effect after commenting...