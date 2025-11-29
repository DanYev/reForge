from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "reforge.rfgmath.rcymath64",        # 64-bit (double) version
        ["reforge/rfgmath/rcymath64.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-fopenmp", "-ffast-math", "-ftree-vectorize", 
                           "-march=native", "-fopt-info-vec-optimized"],  
        extra_link_args=["-fopenmp"]
    ),
    Extension(
        "reforge.rfgmath.rcymath32",        # 32-bit (float) version
        ["reforge/rfgmath/rcymath32.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-fopenmp", "-ffast-math", "-ftree-vectorize",
                           "-march=native", "-fopt-info-vec-optimized"],  
        extra_link_args=["-fopenmp"]
    )
]

setup(
    name="reforge",
    version="0.1.1",
    packages=find_packages(),
    ext_modules=cythonize(extensions),
)

# python setup.py build_ext --inplace