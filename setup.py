from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "reforge.rfgmath.rcmath",           # Full module name
        ["reforge/rfgmath/rcmath.pyx"],     # Path to your .pyx file
        include_dirs=[np.get_include()],     # Include numpy headers if needed
        extra_compile_args=["-O3"],
    )
]

setup(
    name="reforge",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=cythonize(extensions),
)

# python setup.py build_ext --inplace