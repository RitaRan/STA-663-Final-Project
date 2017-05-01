from setuptools import setup
from Cython.Build import cythonize
setup(name='fastfsr',
      version='0.1',
      description='Fast FSR Variable Selection',
      url='https://github.com/CeciliaShi/STA-663-Final-Project',
      author='Xilin Cecilia Shi, Ran Zhou',
      author_email='xs41@duke.edu,rz69@duke.edu',
      license='MIT',
      packages=['fastfsr'],
      ext_modules = cythonize("helper.pyx"),
      zip_safe=False)