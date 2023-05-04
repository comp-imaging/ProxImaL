project('halide toolchain', 'cpp',
    version: '10.0.0')

cc = meson.get_compiler('cpp', native: true)

halide_inc = include_directories('include')

if build_machine.system() == 'windows'
    halide_library_path = meson.current_source_dir() / 'lib/Release'
else
    halide_library_path = meson.current_source_dir() / 'lib'
endif
halide_lib = cc.find_library('Halide', dirs: halide_library_path)

halide_generator_dep = declare_dependency(
    sources: [
    'share/Halide/tools/GenGen.cpp',
    ],
    dependencies: halide_lib, 
    include_directories: halide_inc,
    compile_args: '-fno-rtti',
    #native: true,
)

if not meson.is_cross_build()
    cxx = meson.get_compiler('cpp')

    if build_machine.system() != 'windows'
        dl_lib = cxx.find_library('dl')
    else
        dl_lib = []
    endif

    halide_runtime_dep = declare_dependency(
        include_directories: [
            'include',
            'share/Halide/tools',
        ],
        dependencies: [
            dependency('libpng', required: false),
            dependency('libjpeg', required: false),
            dependency('threads'),
            dl_lib,
        ],
    )
endif
