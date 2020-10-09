from __future__ import print_function

# Includes
import importlib
import subprocess
import os
import sys
import typing
import numpy as np


class Halide(object):

    def __init__(self,
                 func: str,
                 target_shape=(2048, 2048),
                 builddir=[],
                 recompile=False,
                 reconfigure=False,
                 target='host',
                 verbose=False):
        """ Compiles and runs a halide pipeline defined in a generator file ``filepath``
            If recompile is not enabled, first, the library is searched and then loaded.
            Otherwise it is recompiled and a new library is defined.

        Example:

            #Only compiles the source --> generates conv.so file
            Halide('A_conv')

            #Compiles the source and runs function generator conv --> result is in
            numpy array output

            Halide('A_conv').run(A,K,output)

            or

            Halide('A_conv').A_conv(A,K,output) --> calls run
        """

        # Use script location/build as default build directory
        self.cwd = os.path.dirname(os.path.realpath(__file__))
        if not builddir:
            builddir = os.path.join(self.cwd,
                                    'build')

        # Save arguments
        self.func = func
        self.module_name = 'lib{}'.format(func)
        self.builddir = builddir
        self.recompile = recompile
        self.reconfigure = reconfigure
        self.target = target
        self.target_shape = target_shape
        self.verbose = verbose

        # Recompile if necessary
        if self.recompile:
            self.configure()
            self.compile()

        # Syntactic sugar --> create method with name function_name
        setattr(self, self.func, lambda *args: self.run(*args))

    def configure(self):

        is_configured = os.path.exists('{}/build/build.ninja'.format(self.cwd))

        # Force reconfigure
        if self.reconfigure and is_configured:

            subprocess.check_call(['meson', 'configure',
                '-Dhtarget={}'.format(self.target_shape[0]),
                '-Dwtarget={}'.format(self.target_shape[1]),
                self.builddir])

        # Don't need to reconfigure if ninja file exists
        elif is_configured:
            return

        # Default is to setup ninja
        subprocess.check_call(['meson', 'setup',
                '-Dhtarget={}'.format(self.target_shape[0]),
                '-Dwtarget={}'.format(self.target_shape[1]),
                 self.builddir, self.cwd])

    def compile(self):
        ''' Generate the new code (exits on fail) '''

        subprocess.check_call(
            ['ninja', '-C', self.builddir, '{}.so'.format(self.module_name)])

    def run(self, *args):
        """ Execute Halide code that was compiled before. """

        launch = importlib.import_module(
            'proximal.halide.build.{}'.format(self.module_name))
        error = launch.run(*args)

        if error != 0:
            raise RuntimeError('Halide call to {0} returned {2}'.format(
                function_name_c, error)
                  )

class Params:
    """ Supported Params. """
    ImageParam_Float32 = ('ImageParam_Float32', np.float32)
    Param_Float32 = ('Param_Float32', np.float32)
    Param_Int32 = ('Param_Int32', np.int32)
