from distutils.core import setup
from distutils.extension import Extension
import os

if (os.environ.has_key('USE_CYTHON')):
    USE_CYTHON = int(os.environ['USE_CYTHON'])
else:
    USE_CYTHON = False

ext = '.pyx' if USE_CYTHON else '.c'

extensions = [Extension("pyparticleest/utils/kalman", ["pyparticleest/utils/kalman" + ext]),
              Extension("pyparticleest/models/mlnlg_compute", ["pyparticleest/models/mlnlg_compute" + ext])]

if USE_CYTHON:
    from Cython.Build import cythonize
    extensions = cythonize(extensions)

setup(name='pyParticleEst',
      version='1.0',
      packages=['pyparticleest', 'pyparticleest/models', 'pyparticleest/paramest', 'pyparticleest/utils'],
      url='http://www.control.lth.se/Staff/JerkerNordh/pyparticleest.html',
      author='Jerker Nordh',
      author_email='jerker.nordh@control.lth.se',
      description='Framework for particle based estimation methods, such as particle filtering and smoothing',
      license='LGPL',
      ext_modules=extensions
      )
