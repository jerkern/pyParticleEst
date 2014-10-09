from distutils.core import setup
from distutils.extension import Extension
import os

if (os.environ.has_key('USE_CYTHON')):
    USE_CYTHON = int(os.environ['USE_CYTHON'])
else:
    USE_CYTHON = False

ext = '.pyx' if USE_CYTHON else '.c'

extensions = [Extension("pyparticleest/utils/ckalman", ["pyparticleest/utils/ckalman" + ext]),
              Extension("pyparticleest/utils/cmlnlg_compute", ["pyparticleest/utils/cmlnlg_compute" + ext])]

name='pyParticleEst'
version='1.0'
packages=['pyparticleest', 'pyparticleest/models', 'pyparticleest/paramest', 'pyparticleest/utils']
url='http://www.control.lth.se/Staff/JerkerNordh/pyparticleest.html'
author='Jerker Nordh'
author_email='jerker.nordh@control.lth.se'
description='Framework for particle based estimation methods, such as particle filtering and smoothing'
license='LGPL'

if USE_CYTHON:
    from Cython.Build import cythonize
    extensions = cythonize(extensions)

try:
	setup(name=name,
	      version=version,
	      packages=packages,
	      url=url,
	      author=author,
	      author_email=author_email,
	      description=description,
	      license=license,
	      ext_modules=extensions,
      	)
except SystemExit: 
	setup(name=name,
	      version=version,
	      packages=packages,
	      url=url,
	      author=author,
	      author_email=author_email,
	      description=description,
	      license=license,
      	)
