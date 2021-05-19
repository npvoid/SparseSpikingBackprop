from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='s3gd_cuda',
    ext_modules=[
        CUDAExtension('s3gd_cuda', [
            's3gd_cuda.cpp',
            's3gd_cuda_kernel.cu',
        ],

        ),
        # extra_compile_args={'cxx': [], 'nvcc': ['--ptxas-options=-v']}),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
