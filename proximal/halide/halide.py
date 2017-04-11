from __future__ import print_function

# Includes
import subprocess
import os
from ctypes import cdll
import ctypes
import sys
import numpy as np


def halide_installed():
    """Returns whether Halide is installed.
    """
    return 'HALIDE_PATH' in os.environ


class Halide(object):

    def __init__(self, generator_source=[], func=[],
                 builddir=[], recompile=False, target='host',
                 generator_name=[], generator_param=[],
                 external_source=[], external_libs=[],
                 compile_flags=[], cleansource=True, verbose=False):
        """ Compiles and runs a halide pipeline defined in a generator file ``filepath``
            If recompile is not enabled, first, the library is searched and then loaded.
            Otherwise it is recompiled and a new library is defined.

        Example:

            #Only compiles the source --> generates conv.so file
            Halide('conv.cpp')

            #Compiles the source and runs function generator conv --> result is in
            numpy array output

            Halide('conv.cpp').run(A,K,output)

            or

            Halide('conv.cpp').conv(A,K,output) --> calls run

            #Only runs the precompiled halide function --> result is in numpy array output
            Halide('conv.cpp').conv(A,K,output)

        """

        # Use script location/build as default build directory
        if not builddir:
            builddir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'build')

        # Save arguments
        self.generator_source = find_source(generator_source)  # Try to find source file
        self.func = func
        self.builddir = builddir
        self.recompile = recompile
        self.target = target
        self.generator_name = generator_name
        self.generator_param = generator_param
        self.external_source = external_source
        self.external_libs = external_libs
        self.compile_flags = compile_flags
        self.cleansource = cleansource
        self.verbose = verbose

        # Output names that our halide compilation produces
        function_name, function_name_c, output_lib = output_names(
            self.func, self.generator_source, self.builddir)

        # Recompile if necessary
        if not os.path.exists(output_lib) or self.recompile:
            self.compile()

        # Syntactic sugar --> create method with name function_name
        setattr(self, function_name, lambda *args: self.run(*args))

    def compile(self):

        # Generate the new code (exits on fail)
        gengen(self.generator_source, self.builddir,
               self.target, self.generator_name, self.func, self.generator_param,
               self.external_source, self.external_libs, self.compile_flags,
               self.cleansource, self.verbose)

    def run(self, *args):
        """ Execute Halide code that was compiled before. """

        # Output names that our halide compilation produces
        function_name, function_name_c, output_lib = output_names(
            self.func, self.generator_source, self.builddir)

        # Run
        args_ctype = convert_to_ctypes(args, function_name_c)

        # Directly load the lib
        launch = cdll.LoadLibrary(output_lib)

        # Call
        error = getattr(launch, function_name_c)(*args_ctype)

        if error != 0:
            print('Halide call to {0} returned {1}'.format(function_name_c, error), file=sys.stderr)
            exit()


class Params:
    """ Supported Params. """
    ImageParam_Float32 = ('ImageParam_Float32', np.float32)
    Param_Float32 = ('Param_Float32', np.float32)
    Param_Int32 = ('Param_Int32', np.int32)


def output_names(function_name, generator_source, builddir):

    # Define function name
    if not function_name:
        function_name = os.path.splitext(os.path.basename(generator_source))[0]

    if not function_name:
        print('Could not determine function name.', file=sys.stderr)
        exit()

    # Function name in c-launcher
    function_name_c = '{0}_c'.format(function_name)

    # Output library
    lib_name = os.path.join(builddir, function_name + '.so')

    return function_name, function_name_c, lib_name


def find_source(source):

    source_found = source
    if not os.path.exists(source_found):

        # Try with path substitution
        sourcedir = os.path.dirname(source)
        sourcebase = os.path.basename(source)
        halide_dir = os.path.dirname(os.path.realpath(__file__))
        source_found = os.path.join(halide_dir, 'src', sourcebase)

        if not os.path.exists(source_found):

            # Try without path substitution
            halide_dir = os.path.dirname(os.path.realpath(__file__))
            source_found = os.path.join(halide_dir, 'src', source)

            if not os.path.exists(source_found):
                print('Error: Could not find source {0} in {1} or in {2}'.format(
                    sourcebase, sourcedir, halide_dir), file=sys.stderr)
                exit()

    return source_found


def gengen(generator_source, builddir='./build',
           target='host', generator_name=[], function_name=[], generator_param=[],
           external_source=[], external_libs=[], compile_flags=[],
           cleansource=True, verbose=True):
    """ Will take .cpp containing one (or more) Generators, compile them,
        link them with libHalide, and run
        the resulting executable to produce a .o/.h expressing the Generator's
        function. Function name is the C function name for the result """

    # Build directory
    if not os.path.exists(builddir):
        os.makedirs(builddir)

    # Generator code is a temporary file
    generator = os.path.join(builddir, 'gengen.XXXX')

    # File definitions
    halide_lib = '${HALIDE_PATH}/bin/libHalide.so'
    halid_incl = '-I${HALIDE_PATH}/include'
    generator_main = '${HALIDE_PATH}/tools/GenGen.cpp'

    # Define output names
    function_name, function_name_c, output_lib = output_names(
        function_name, generator_source, builddir)

    # It's OK for GENERATOR_NAME and FUNCTION_NAME to be empty
    # if the source we're compiling has only one generator registered,
    # we just use that one (and assume that FUNCTION_NAME=GENERATOR_NAME)
    generator_flag = ""
    if generator_name:
        generator_flag = "-g " + generator_name

    # Function flag
    function_flag = "-f " + function_name

    # Target flags
    target_flags = "target=" + target

    launcher_file = ''

    try:

        # Additional flags
        compile_flag_str = ''
        if compile_flags:
            for cf in compile_flags:
                compile_flag_str += cf + ' '

        # Compile
        cmd = ("g++ {0} -g -Wwrite-strings -std=c++11 -fno-rtti {1} {2} {3} {4} "
               " -lz -lpthread -ldl -o {5}").format(
            compile_flag_str, halid_incl, generator_source, generator_main, halide_lib, generator)

        if verbose:
            print('Compiling {0}'.format(generator_source))
            print('\t' + cmd)
        subprocess.call(cmd, shell=True)

        # Run generator
        cmd = '{0} {1} {2} -e o,h -o {3} {4}'.format(generator,
                                                     generator_flag, function_flag,
                                                     builddir, target_flags)
        if verbose:
            print('Calling generator')
            print('\t' + cmd)
        subprocess.call(cmd, shell=True)

        # Find params in output generated by generator
        header_file = os.path.join(builddir, function_name + '.h')
        object_file = os.path.join(builddir, function_name + '.o')
        params = scan_params(header_file, function_name, verbose)
        if verbose:
            print('Found {0} buffers and {1} float params and {2} int params'.format(
                params.count(Params.ImageParam_Float32),
                params.count(Params.Param_Float32),
                params.count(Params.Param_Int32)))

        # Generate launcher cpp and write
        launcher_file = os.path.join(builddir, function_name + '.cpp')
        launcher_body, argument_names = generate_launcher(
            header_file, function_name, function_name_c, params)
        with open(launcher_file, 'w') as fp:
            fp.write(launcher_body)

        # Compile launcher into library file (which will be called later by ctypes)
        if os.path.exists(output_lib):
            os.remove(output_lib)

        # External sources
        external_source_str = ''
        if external_source:
            for sc in external_source:
                external_source_str += find_source(sc) + ' '

        external_libs_str = ''
        if external_libs:
            for el in external_libs:
                external_libs_str += el + ' '

        cmd = ("g++ -fPIC -std=c++11 -Wall -O2 {0} {1} {2} -lpthread "
               "{3} -shared -o {4}").format(launcher_file, external_source_str,
                                            external_libs_str, object_file, output_lib)
        if verbose:
            print('Compiling library')
            print('\t' + cmd)
        subprocess.call(cmd, shell=True)

        return output_lib, function_name_c, argument_names

    except Exception as e:
        print('Error genererator compilation: {0}'.format(e.message), file=sys.stderr)
        exit()

    finally:
        # Cleanup
        if cleansource:
            source = [generator, header_file, object_file, launcher_file]
            for selem in source:
                if os.path.exists(selem):
                    os.remove(selem)


def convert_to_ctypes(args, func):
    """ Converts an argument list to a ctype compatible list for our launcher function """

    # pass numpy buffers using ctypes
    cargs = []
    try:

        # Iterate over the args and convert
        for arg in args:

            # Check for supported types
            if isinstance(arg, np.ndarray):

                # Check for the array type
                if not arg.dtype == np.float32:
                    raise ValueError(
                        'Input array of type {0} detected, Not supported.'.format(arg.dtype))

                # Otherwise add the bounds
                if len(arg.shape) > 4:
                    raise ValueError(
                        'Detected {0} dimensions. Halide supports only up to 4.'.format(
                            len(arg.shape)))

                # Check if fortran array
                if len(arg.shape) > 1 and not np.isfortran(arg):
                    print('Arg ', arg)
                    # Much faster and more natural halide code
                    raise ValueError('Currently supports only Fortran order')

                # Add ctype
                cargs.append(arg.ctypes.data_as(ctypes.c_void_p))

                # Add bound w,h,x,y ...
                for s in [1, 0, 2, 3]:
                    if s < len(arg.shape):
                        cargs.append(ctypes.c_int(np.int32(arg.shape[s])))
                    else:
                        cargs.append(ctypes.c_int(np.int32(1)))

            elif isinstance(arg, float) or isinstance(arg, np.float32):
                cargs.append(ctypes.c_float(arg))

            elif isinstance(arg, int) or isinstance(arg, np.int32):
                cargs.append(ctypes.c_int(arg))

            else:
                raise ValueError('Unsupported type.')

    except Exception as e:
        print('Error argument conversion: {0} in func {1}'.format(e.message, func), file=sys.stderr)
        exit()

    return cargs


def scan_params(header_file, function_name, verbose=True):
    """ Scans a header file from generator and reads params from compiled fucntion
    """

    # Find signature
    signature_start = 'int {}('.format(function_name)
    signature = ""
    searchfile = open(header_file, "r")
    for line in searchfile:
        if signature_start in line:
            signature = line
    searchfile.close()

    # Verbose
    if verbose:
        print('Signature found: {}'.format(signature))

    # Find the argumetn list from the signature
    # Assumes halide checks internally if the arguments are correct (e.g.
    # buffer_t with wrong elem size)
    arglist = []
    for arg_string in signature.split(","):
        if 'buffer_t *' in arg_string:
            arglist.append(Params.ImageParam_Float32)
        elif 'float' in arg_string:
            arglist.append(Params.Param_Float32)
        elif 'int' in arg_string:
            arglist.append(Params.Param_Int32)
        else:
            print('Error: Unsupported argument in {0}'.format(arg_string), file=sys.stderr)

    # Check for buffer count
    if arglist.count(Params.ImageParam_Float32) == 0:
        print('Error: No input buffers in halide generated header {0}'.format(
            header_file), file=sys.stderr)

    return arglist


def generate_launcher_arguments(params):

    # Generates buffer definitions for the launcher
    # Example
    # buffer_t b_input = {0, (uint8_t*)input, {width,height}, {1,width}, {0}, sizeof(float),};

    # The parameters are just numbered
    buffer_defs = []
    argument_defs = []
    argument_names = []
    call_names = []
    for id, param in enumerate(params):

        # Name of current argument
        argname = 'arg{:d}'.format(id)
        argument_names.append(argname)

        # Create buffer if buffer argument
        if param == Params.ImageParam_Float32:
            buffer_name = 'buf{0}'.format(argname)
            bbody = 'buffer_t {0}'.format(buffer_name)
            bbody += ' = {{0, (uint8_t*){0}, '.format(argname)  # Float pointer

            # Linear strides
            # bbody += '{{ {0}w,{0}h,{0}s,{0}t }}, {{ 1, {0}w, {0}w * {0}h, {0}w *
            # {0}h * {0}s }}, '.format(argname) #Extent and stride

            # Flip first two axis for natural buffer access (numpy like)
            # Extent and stride
            bbody += ("{{ {0}w,{0}h,{0}s,{0}t }}, "
                      "{{ {0}h, 1, {0}w * {0}h, {0}w * {0}h * {0}s }}, ").format(
                argname)

            bbody += '{0}, sizeof(float),};'

            # Add to buffer definitions
            buffer_defs.append(bbody)

            # Add argument and bounds
            argument_defs.append(
                'float* {0}, int {0}w, int {0}h, int {0}s, int {0}t'.format(argname))
            call_names.append('&{0}'.format(buffer_name))

        elif param == Params.Param_Float32:

            # Add argument
            argument_defs.append('const float {0}'.format(argname))
            call_names.append(argname)

        elif param == Params.Param_Int32:
            # Add argument
            argument_defs.append('const int {0}'.format(argname))
            call_names.append(argname)

    return argument_names, argument_defs, buffer_defs, call_names


def generate_launcher(header_file, function_name, function_name_c, params):
    """ Generates launcher glue code that runs generator from c-interface.
        Similar to matlab runtime in halide. """

    # Header
    body = '// Launcher code for generator of function: {0}\n'.format(function_name)
    body += ("#include <stdio.h>\n#include <stdlib.h>\n#include "
             """"{0}.h"\n\n#define LIBRARY_API extern "C"\n\n""").format(
        function_name)

    # Generate arguments
    argument_names, argument_defs, buffer_defs, call_names = generate_launcher_arguments(params)

    # Method signature
    function_def = 'LIBRARY_API int {0}('.format(function_name_c)
    body += function_def  # Contains at least one buffer
    for aid, argdef in enumerate(argument_defs):
        if aid == 0:
            body += argdef
        else:
            body += '\n' + (' ' * len(function_def)) + argdef
        body += ',' if aid < len(argument_defs) - 1 else ''  # Trailing comma

    # Function body
    body += ')\n{\n'
    INDENT = '    '

    # Add a buffer_t for each buffer
    body += INDENT + '//Transfer data from arrays into buffers\n'
    for buf in buffer_defs:
        body += INDENT + buf + '\n'
    body += '\n'

    # Arguments for the call
    call_argument = ''
    for cid, cn in enumerate(call_names):
        call_argument += cn
        call_argument += ', ' if cid < len(call_names) - 1 else ''  # Trailing comma

    # Call the function
    body += INDENT + '//Call our halide function with the buffers defined\n'
    body += INDENT + '{0}( {1} );\n\n'.format(function_name, call_argument)

    # Return
    body += INDENT + '// DONE\n'
    body += INDENT + 'return 0;\n'
    body += '}\n'

    return body, argument_names
