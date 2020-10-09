project('proximal-halide', 'cpp',
    default_options: [
        'buildtype=debugoptimized',
])

halide_toolchain = subproject('halide')
halide_generator_dep = halide_toolchain.get_variable('halide_generator_dep')
halide_runtime_dep = halide_toolchain.get_variable('halide_runtime_dep')

pipeline_src = [
    'src/A_conv.cpp',
    'src/At_conv.cpp',
    'src/prox_L1.cpp',
    'src/prox_IsoL1.cpp',
    'src/fft2_r2c.cpp',
    'src/ifft2_c2r.cpp',
    'src/fft/fft.cpp',
    'src/At_grad.cpp',
    'src/A_grad.cpp',
    'src/A_mask.cpp',
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
]]

python_dep = dependency('python3')
pybind11_dep = subproject('pybind11').get_variable('pybind11_dep')

foreach p : pipeline_name
    obj = custom_target(
        p[0] + '.o',
        output: [
            p[0] + '.o',
            p[0] + '.h',
        ],
        input: halide_generator,
        command: [
            halide_generator,
            '-o', meson.current_build_dir(),
            '-g', p[0],
            '-e', 'o,h',
            'target=host',
            'auto_schedule=@0@'.format(p[2]),
            'machine_params=8,6291000,40',
            p[3],
        ],
    )

    foreach library_name : p[1]
        # TODO: use meson python extension_module to infer the suffix
        # of the library
        #lib = py.extension_module(
        lib = library(
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
    endforeach
endforeach