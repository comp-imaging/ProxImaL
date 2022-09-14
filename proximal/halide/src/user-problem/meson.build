user_problem_dep = declare_dependency(
    include_directories: '.',
    dependencies: [
        core_dep,
        ladmm_dep,
    ],
)

solver_generator = executable(
    'solver-generator',
    sources: [
        'linearized-admm-gen.cpp',
    ],
    dependencies: [
        halide_generator_dep,
        ladmm_dep,
        user_problem_dep,
    ],
)

solver_bin = custom_target(
    'ladmm_iter.a',
    output: [
        'ladmm_iter.a',
        'ladmm_iter.h',
    ],
    input: solver_generator,
    env: {
        'LD_LIBRARY_PATH': halide_library_path,
    },
    command: [
        solver_generator,
        '-o', meson.current_build_dir(),
        '-g', 'ladmm_iter',
        '-e', 'static_library,h,stmt_html',
        'machine_params=12,6291000,40',
        'target=host',
        'auto_schedule=true',
        '-s','Mullapudi2016',
        '-p','autoschedule_mullapudi2016',

        'n_iter=1',     # number of ADMM iterations before checking convergence
        'mu=0.333',     # Problem scaling factor. Defaults to 1 / sqrt( || K || ).
        'lmb=3.0',      # Problem scaling factor. Defaults to sqrt( || K || ).
    ],
    build_by_default: true,
)

test_runtime_exe = executable('test-runtime',
    sources: [
        # TODO(Antony): to separate the ADMM runtime library from the test code.
        'test-runtime.cpp',
        solver_bin,
    ],
    cpp_args: [
        '-DRAW_IMAGE_PATH="@0@"'.format(parrot_img),
    ],
    dependencies: [
        halide_runtime_dep,
        dependency('libpng'),
    ],
)

test('Halide-optimized L-ADMM with Proximal-Codegen',
    test_runtime_exe,
    is_parallel: false,
)