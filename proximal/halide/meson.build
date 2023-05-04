project('proximal-halide', 'cpp',
    meson_version: '>=0.58.1',
    default_options: [
        'buildtype=debugoptimized',
        'cpp_std=c++17',
])

halide_toolchain = subproject('halide-@0@-@1@'.format(build_machine.cpu_family(), build_machine.system()))
halide_generator_dep = halide_toolchain.get_variable('halide_generator_dep')
halide_runtime_dep = halide_toolchain.get_variable('halide_runtime_dep')
halide_library_path = halide_toolchain.get_variable('halide_library_path')

pipeline_src = [
    'src/A_conv.cpp',
    'src/At_conv.cpp',
    'src/prox_L1.cpp',
    'src/prox_IsoL1.cpp',
    'src/prox_Poisson.cpp',
    'src/fft2_r2c.cpp',
    'src/ifft2_c2r.cpp',
    'src/fft/fft.cpp',
    'src/At_grad.cpp',
    'src/A_grad.cpp',
    'src/A_mask.cpp',
    'src/A_warp.cpp',
    'src/At_warp.cpp',
]

halide_generator = executable(
    'halide_pipelines',
    sources: pipeline_src,
    dependencies: [
        halide_generator_dep,
    ],
)

generator_param = [
  'wtarget=@0@'.format(get_option('wtarget')),
  'htarget=@0@'.format(get_option('htarget')),
]

pipeline_name = [[
        'convImg',
        ['A_conv'],
        true,
        [],
    ], [
        'convImgT',
        ['At_conv'],
        true,
        [],
    ], [
        'proxL1',
        ['prox_L1'],
        true,
        [],
    ], [
        'proxIsoL1',
        ['prox_IsoL1'],
        true,
        [],
    ], [
        'proxPoisson',
        ['prox_Poisson'],
        true,
        [],
    ], [
        'fftR2CImg',
        ['fft2_r2c'],
        false,
        generator_param,
    ], [
        'ifftC2RImg',
        ['ifft2_c2r'],
        false,
        generator_param,
    ], [
        'gradTransImg',
        ['At_grad'],
        true,
        [],
    ], [
        'gradImg',
        ['A_grad'],
        true,
        [],
    ], [
        'WImg',
        ['A_mask', 'At_mask'],
        true,
        [],
    ], [
        'warpImg',
        ['A_warp'],
        false,
        [],
    ], [
        'warpImgT',
        ['At_warp'],
        false,
        [],
]]

py = import('python').find_installation()
python_dep = py.dependency()
pybind11_dep = subproject('pybind11').get_variable('pybind11_dep')

cuda_toolchain = find_program('nvcc', required: false)

if build_machine.system() == 'windows'
    env = { 'PATH': halide_library_path }
    object_file_ext = 'obj'
    statlib_file_ext = 'lib'
else
    env = { 'LD_LIBRARY_PATH': halide_library_path }
    object_file_ext = 'o'
    statlib_file_ext = 'a'
endif

foreach p : pipeline_name
    compile_cmd = [
        halide_generator,
        '-o', meson.current_build_dir(),
        '-g', p[0],
        '-e', 'o,h',
        'machine_params=4,6291000,40',
    ]

    auto_schedule = p[2]

    if cuda_toolchain.found() and auto_schedule
        compile_cmd += [
            'target=host-cuda',
            'auto_schedule=true',
            '-s','Li2018',
            '-p','autoschedule_li2018',
        ]
    elif auto_schedule
        compile_cmd += [
            'target=host',
            'auto_schedule=true',
            '-s','Mullapudi2016',
            '-p','autoschedule_mullapudi2016',
        ]
    else
        compile_cmd += [
            'target=host',
            'auto_schedule=false',
        ]
    endif

    obj = custom_target(
        p[0] + '.[oh]',
        output: [
            p[0] + '.' + object_file_ext,
            p[0] + '.h',
        ],
        input: halide_generator,
        env: env,
        command: [
            compile_cmd,
            p[3],
        ],
    )

    foreach library_name : p[1]
        lib = py.extension_module(
            library_name,
            sources: [
                'interface/@0@.cpp'.format(library_name),
                obj,
            ],
            dependencies: [
                python_dep,
                pybind11_dep,
                halide_runtime_dep,
            ],
        )

        alias_target(library_name, lib)
    endforeach
endforeach

subdir('src/test_data')
subdir('src/core')
subdir('src/algorithm')
subdir('src/user-problem')