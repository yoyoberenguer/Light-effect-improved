# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [Extension("SATURATION", ["Saturation.pyx"],
                         include_dirs=[numpy.get_include()],
                         extra_compile_args=['/openmp'],
                         extra_link_args=['/openmp'],
                         ),
               Extension("MAPPING", ["Mapping.pyx"],
                         include_dirs=[numpy.get_include()],
                         extra_compile_args=['-openmp'],
                         extra_link_args=['-openmp'],
                         )
               ]

setup(
  name="SATURATION",
  cmdclass={"build_ext": build_ext},
  ext_modules=ext_modules,
  include_dirs=[numpy.get_include()]
)
