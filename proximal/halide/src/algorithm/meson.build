range_dep = subproject('range-v3').get_variable('range_dep')

ladmm_dep = declare_dependency(
    include_directories: '.',
    dependencies: [
        halide_generator_dep,
        range_dep,
    ],
)