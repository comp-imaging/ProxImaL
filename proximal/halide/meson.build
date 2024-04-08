project('proximal-halide', 'cpp',
    meson_version: '>=0.60.0',
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
    'src/least_square_direct.cpp',
    'src/fft/fft.cpp',
    'src/At_grad.cpp',
    'src/A_grad.cpp',
    'src/A_mask.cpp',
    'src/A_warp.cpp',
    'src/At_warp.cpp',
]

if get_option('build_nlm')
    pipeline_src += 'src/prox_NLM.cpp'
endif

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

pipeline_name = [{
        'name': 'convImg',
        'interfaces': ['A_conv'],
        'autoschedule': true,
    }, {
        'name': 'convImgT',
        'interfaces': ['At_conv'],
        'autoschedule': true,
    }, {
        'name': 'proxL1',
        'interfaces': ['prox_L1'],
        'autoschedule': true,
    }, {
        'name': 'proxIsoL1',
        'interfaces': ['prox_IsoL1'],
        'autoschedule': true,
    }, {
        'name': 'proxPoisson',
        'interfaces': ['prox_Poisson'],
        'autoschedule': true,
    }, {
        'name': 'fftR2CImg',
        'interfaces': ['fft2_r2c'],
        'autoschedule': false,
        'generator_param': generator_param,
    }, {
        'name': 'least_square_direct',
        'interfaces': ['prox_L2'],
        'autoschedule': false,
        'generator_param': generator_param,
    }, {
        'name': 'least_square_direct',
        'function_name': 'least_square_direct_ignore_offset',
        'interfaces': ['prox_L2_ignore_offset'],
        'autoschedule': false,
        'generator_param': generator_param + 'ignore_offset=true',
    }, {
        'name': 'ifftC2RImg',
        'interfaces': ['ifft2_c2r'],
        'autoschedule': false,
        'generator_param': generator_param,
    }, {
        'name': 'gradTransImg',
        'interfaces': ['At_grad'],
        'autoschedule': true,
    }, {
        'name': 'gradImg',
        'interfaces': ['A_grad'],
        'autoschedule': true,
    }, {
        'name': 'WImg',
        'interfaces': ['A_mask', 'At_mask'],
        'autoschedule': true,
    }, {
        'name': 'warpImg',
        'interfaces': ['A_warp'],
        'autoschedule': false,
    }, {
        'name': 'warpImgT',
        'interfaces': ['At_warp'],
        'autoschedule': false,
}]

py = import('python').find_installation()
python_dep = py.dependency()
pybind11_dep = subproject('pybind11').get_variable('pybind11_dep')

cuda_toolchain = find_program('nvcc', required: false)

if get_option('build_nlm')
    # Provides libnlm_extern.so
    subdir('src/external')

    pipeline_name += {
        'name': 'proxNLM',
        'interfaces': ['prox_NLM'],
        'autoschedule': false,
        'link_with': nlm_extern_lib,
    }
endif

if build_machine.system() == 'windows'
    env = { 'PATH': [
        halide_toolchain.get_variable('halide_dll_path'),
        halide_library_path,
    ]}
    object_file_ext = 'obj'
    statlib_file_ext = 'lib'
else
    env = { 'LD_LIBRARY_PATH': halide_library_path }
    object_file_ext = 'o'
    statlib_file_ext = 'a'
endif

proximal_python_interface = []

foreach p : pipeline_name
    if not p.has_key('function_name')
        p += {'function_name': p['name']}
    endif

    compile_cmd = [
        halide_generator,
        '-o', meson.current_build_dir(),
        '-g', p['name'],
        '-e', 'o,h',
        '-f', p['function_name'],
    ]

    if cuda_toolchain.found() and p['autoschedule']
        compile_cmd += [
            'target=host-cuda',
            '-p', 'autoschedule_li2018',
            'autoscheduler=Li2018',
            'autoscheduler.parallelism=32',
        ]
    elif p['autoschedule']
        compile_cmd += [
            'target=host',
            '-p', 'autoschedule_mullapudi2016',
            'autoscheduler=Mullapudi2016',
            # Maximum level of CPU core, or GPU threads available
            'autoscheduler.parallelism=4',
            # Size of last level (L2) cache
            'autoscheduler.last_level_cache_size=6291000',
            # Ratio of the cache read cost to compute cost
            'autoscheduler.balance=40',
        ]
    else
        compile_cmd += [
            'target=host',
        ]
    endif

    if not p.has_key('generator_param')
        p += {'generator_param': []}
    endif

    obj = custom_target(
        p['function_name'] + '.[oh]',
        output: [
            p['function_name'] + '.' + object_file_ext,
            p['function_name'] + '.h',
        ],
        input: halide_generator,
        env: env,
        command: [
            compile_cmd,
            p['generator_param'],
        ],
    )

    if not p.has_key('link_with')
        p += {'link_with': []}
    endif

    foreach library_name : p['interfaces']
        lib = py.extension_module(
            library_name,
            sources: [
                'interface/@0@.cpp'.format(library_name),
                obj,
            ],
            cpp_args: [
                '-fvisibility=hidden',
            ],
            link_with: p['link_with'],
            dependencies: [
                python_dep,
                pybind11_dep,
                halide_runtime_dep,
            ],
        )

        proximal_python_interface += lib

        alias_target(library_name, lib)
    endforeach
endforeach

subdir('src/test_data')
subdir('src/core')
subdir('src/algorithm')
subdir('src/user-problem')

alias_target('python_interface', proximal_python_interface)