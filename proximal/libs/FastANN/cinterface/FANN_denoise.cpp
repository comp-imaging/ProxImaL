//Includes
#include "FANN_denoise.h"
#include "gpuinterface.h"
#include "paramloader.h"
#include <ip/Image.h>
#include <sstream>
#include <iostream>
#include <utils/Utils.h>
#include <math.h>

void run_FANN_denoise(float* input, int width, int height, int channels, 
                      float sigma, float* params, int numParams, bool verbose, 
                      float* output, float* stats)
{
    //Check for channel
    if( channels != 1 && channels != 3)
    {
        std::cerr << "input image must either have one channel (greyscale) or " 
                  << "three channels (RGB) but has " << channels << std::endl;
        return;
    }

    //Parse image data
    std::vector<Image<IPF_float32> > inputImage;
    NlmChannelInfo supposedFormat = NLM_CI_RGB;
    if(channels == 1)
    {
        supposedFormat = NLM_CI_GREYSCALE;
        inputImage.resize(1);
        copyFromArray<IPF_float32>("Greyscale_input", inputImage[0], input, 0, width, height);
    }
    else{
        supposedFormat = NLM_CI_RGB;
        inputImage.resize(3);
        copyFromArray<IPF_float32>("RGB_input", inputImage[0], input, 0, width, height);
        copyFromArray<IPF_float32>("RGB_input", inputImage[1], input, width*height, width, height);
        copyFromArray<IPF_float32>("RGB_input", inputImage[2], input, 2*width*height, width, height);
    }

    //Parse parameter
    Config config;
    config.Algorithm = NLM_DNA_BM3D_WIENER;
    config.Format = supposedFormat;
    config.BlockSize = 8;
    config.TileSize = 15;
    config.ClusterSize = 32;
    config.NumCandidates = 16;

    loadCommonParams(&config, numParams, params, verbose);

    //Check all the options
    if(supposedFormat == NLM_CI_GREYSCALE && config.Format != NLM_CI_GREYSCALE){
        std::cerr << "Only a single input channel is provided, can only use Greyscale." << std::endl;
        return;
    }

    if(supposedFormat == NLM_CI_RGB && 
        (config.Format == NLM_CI_YUV_420 || config.Format == NLM_CI_YUV_420_FAST || config.Format == NLM_CI_YUV_420_FAST_SUBPIXEL) ){
        std::cerr << "Can only use YUV420 when input is already YUV420." << std::endl;
        return;
    }

    if(supposedFormat ==  NLM_CI_RGB && config.Format == NLM_CI_GREYSCALE){
       std::cerr << "Can only use Greyscale when input is already Greyscale." << std::endl;
       return;
    }

    if(supposedFormat == NLM_CI_YUV_420 &&
            !(config.Format == NLM_CI_YUV_420 || config.Format == NLM_CI_YUV_420_FAST || config.Format == NLM_CI_YUV_420_FAST_SUBPIXEL) ){
        std::cerr << "If input is YUV420 only a YUV420 format can be used." << std::endl;
        return;
    }

    //Stats
    float runtime;
    float memusage;

    Channels inputchannels;
    Channels outputchannels;
    nv::FrameBufferManager& fbm = nv::FrameBufferManager::GetInstance();
    for(int i = 0; i < inputImage.size(); ++i)
    {
        inputchannels.push( gpu::IPF2FrameBuffer(inputImage[i]) );
        outputchannels.push(fbm.Create(inputImage[i].getWidth(), inputImage[i].getHeight(), nv::Type2PixType<float>::PixT, PITCH2D, RD_ELEMENT_TYPE)  );
    }

    std::stringstream messages;
    bool success = runDenoiser(config, sigma, inputchannels, outputchannels, memusage, runtime, messages);
    if(messages.str().size() > 1)
        std::cerr << messages.str() << std::endl;
    if(!success)
        std::cerr << "Denoising failed - unknown problem" << std::endl;

    std::vector<Image<IPF_float32> > outputImage(inputImage.size());
    for(int i = 0; i < inputImage.size(); ++i)
        outputImage[i] = gpu::FrameBuffer2IPF<IPF_float32>(outputchannels[i]);

    //Copy result
    if(outputImage.size() == 1)
    {
        for(int y = 0; y < height; ++y)
            for(int x = 0; x < width; ++x)
                output[x*height + y] =  outputImage[0](x,y);
    }
    else if(outputImage[0].getHeight() == outputImage[1].getHeight())
    {
        //single three channel image
        for(int y = 0; y < height; ++y)
            for(int x = 0; x < width; ++x)
            {
                output[0 * width * height + x*height + y] =  outputImage[0](x,y);
                output[1 * width * height + x*height + y] =  outputImage[1](x,y);
                output[2 * width * height + x*height + y] =  outputImage[2](x,y);
            }
    }
   
    if(stats != nullptr)
    {
        stats[0] = runtime/1000.0f;
        stats[1] = memusage;
    }

    //Destroy frame buffer object
    fbm.ShutDown();
    
    return;
}
