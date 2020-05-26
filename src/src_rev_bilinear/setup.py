from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(name='rev_bilinear',
      ext_modules=[CUDAExtension('rev_bilinear', ['rev_bilinear.cpp', 'forward.cu', 'backward.cu'])],
      cmdclass={'build_ext': BuildExtension})
#setup(name='rev_bilinear',
#      ext_modules=[CUDAExtension('rev_bilinear', ['rev_bilinear.cpp', 'forward.cu'])],
#      cmdclass={'build_ext': BuildExtension})

