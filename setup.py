from __future__ import print_function

import warnings
from setuptools import setup, find_packages, Extension
from setuptools.command.install import install

class layer_install(install):
    def run(self):
        print("please type `install`.\n")
        mode = None
        return install.run(self)

cmdclass = {}
ext_modules = []
cmdclass.update({'install': layer_install})

setup(
    cmdclass=cmdclass,
    ext_modules=ext_modules,
    name='layer',
    version="0.1.11",
    author="Pannous",
    author_email="info@pannous.com",
    packages=find_packages(),
    description='tensorflow custom comfort wrapper',
    license='Apache2 license',
    long_description=open('README.md', 'rb').read().decode('utf8'),
    dependency_links=['git+http://github.com/pannous/context.git#egg=layer'],
    install_requires=['tensorflow'],
    # scripts=['bin/angle'],
    package_data={
        # '': ['*.cu', '*.cuh', '*.h'],
    },
)
