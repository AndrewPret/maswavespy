# setup.py
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

ext = Extension(
    name="cy_theoretical_dc",       # name of the compiled module
    sources=["cy_theoretical_dc.pyx"],
    include_dirs=[numpy.get_include()]
)

setup(
    name="cy_theoretical_dc",
    ext_modules=cythonize(ext, compiler_directives={'language_level': "3"}),
)