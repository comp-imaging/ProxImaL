if not cuda_toolchain.found()
    error('cv::fastNlMeanDenoising needs cuda runtime to function.')
endif

nlm_extern_lib = library('nlm_extern',
    sources: [
        'external_NLM.cpp',
    ],
    dependencies: [
        dependency('opencv4'),
    ],
)

alias_target('nlm_extern', nlm_extern_lib)