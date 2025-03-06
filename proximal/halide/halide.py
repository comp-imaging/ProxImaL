# Includes
import importlib
import os
import subprocess

import numpy as np


class Halide:

    def __init__(self,
                 func: str,
                 target_shape: tuple[int, int]=(2048, 2048),
                 builddir: str | None =None,
                 recompile: bool =False,
                 reconfigure: bool =False,
                 target: str ='host',
                 verbose: bool=False):
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
        if builddir is None:
            builddir = os.path.join(self.cwd,
                                    'build')

        # Save arguments
        self.func = func
        self.module_name = func
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

    def configure(self) -> None:

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

    def compile(self) -> None:
        ''' Generate the new code (exits on fail) '''

        subprocess.check_call(
            ['ninja', '-C', self.builddir, self.module_name])

    def run(self, *args: np.ndarray) -> None:
        """ Execute Halide code that was compiled before. """

        launch = importlib.import_module(
            'proximal.halide.build.{}'.format(self.module_name))

        if self.module_name[:4] == 'fft2':
            expected_shape = (launch.htarget, launch.wtarget)
            if np.any(expected_shape != self.target_shape):
                print(f'Warning: FFT2 shape mismatch. Expected {expected_shape}, found {self.target_shape}. Please recompile.')

            if np.any(expected_shape != args[0].shape):
                print('Warning: Input image shape mismatch for FFT2. '
                      f'Expected {expected_shape}, found {args[0].shape}. Applying circular boundary condition.')

        error : int = launch.run(*args)

        if error != 0:
            raise RuntimeError(f'Halide call to {self.func:s} returned {error:d}')

class Params:
    """ Supported Params. """
    ImageParam_Float32 = ('ImageParam_Float32', np.float32)
    Param_Float32 = ('Param_Float32', np.float32)
    Param_Int32 = ('Param_Int32', np.int32)
