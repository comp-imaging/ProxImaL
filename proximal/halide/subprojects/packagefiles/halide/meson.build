project('halide toolchain', 'cpp')

cc = meson.get_compiler('cpp', native: true)

halide_inc = include_directories('include')
halide_lib = cc.find_library('Halide', dirs: join_paths(meson.current_source_dir(), 'bin'))

halide_generator_dep = declare_dependency(
    sources: [
    'tools/GenGen.cpp',
    ],
    dependencies: halide_lib, 
    include_directories: halide_inc,
    compile_args: '-fno-rtti',
)

if not meson.is_cross_build()
    cxx = meson.get_compiler('cpp')

    halide_runtime_dep = declare_dependency(
        include_directories: [
            'include',
            'tools',
        ],
        dependencies: [
            dependency('libpng', required: false),
            dependency('libjpeg', required: false),
            dependency('threads'),
            cxx.find_library('dl'),
        ],
    )
endif
