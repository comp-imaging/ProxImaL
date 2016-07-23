/*
###################################################
###### Denoising mex interface
 */

//Includes
#include "mex.h"
#include "matrix.h"
#include "gpuinterface.h"
#include "paramloader.h"
#include <ip/Image.h>
#include <sstream>
#include <utils/Utils.h>

//Primal Dual Optimization includes
//Math
#include <math.h>

//####################################
//#### Main mex entry point 
//####################################



void mexFunction(int nlhs, mxArray *plhs[], /* Output variables */
		 int nrhs, const mxArray *prhs[]) /* Input variables */
{

    if(nrhs < 2)
    {
        mexPrintf("fastdenoise_mex:\n");
        mexPrintf("  [Greyscale_image info] = fastdenoise_mex(Greyscale_input, noise_sigma, <config_paramname>, <config_value>, ...)\n");
        mexPrintf("  [RGB_image info] = fastdenoise_mex(RGB_input, noise_sigma, <config_paramname>, <config_value>, ...)\n");
        mexPrintf("  [Y_channel U_channel V_channel info] = fastdenoise_mex(Y_channel_input, U_channel_input, V_channel_input, noise_sigma, <config_paramname>, <config_value>, ...)\n");
        mexPrintf("  General named parameters are:\n");
        printCommonParams();
        mexErrMsgTxt("Insufficient number of inputs");
        return;
    }
    if(nlhs < 1)
    {
        mexErrMsgTxt("Insufficient number of outputs");
    }
    if(nlhs > 4)
    {
        mexErrMsgTxt("Too many output parameters");
    }




    NlmChannelInfo supposedFormat = NLM_CI_RGB;

    const int* inputdims = mxGetDimensions(prhs[0]);
    const int* inputdims2 = mxGetDimensions(prhs[1]);

    std::vector<Image<IPF_float32> > inputImage;
    int paramsoffset = 0;
    int width, height;
    float sigma;

    if(mxGetNumberOfDimensions(prhs[0]) > 2 && inputdims[2] != 1 && inputdims[2] != 3)
    {
        mexPrintf("input image must either have one channel (greyscale) or three channels (RGB) but has %d\n", inputdims[2]);
        mexErrMsgTxt("Insufficient input data");
        return;
    }
    width = inputdims[1];
    height = inputdims[0];
    if((mxGetNumberOfDimensions(prhs[0]) == 2 || inputdims[2] == 1) && inputdims2[0] > 1 && inputdims2[1] > 1 && (mxGetNumberOfDimensions(prhs[1]) == 2 || inputdims2[2] == 1))
    {
        if(nrhs < 4)
            mexErrMsgTxt("Not enough input data for separate channel mode: fastdenoise_mex(Y, U, V, noise_sigma)");
        const int* inputdims3 = mxGetDimensions(prhs[2]);
        if(inputdims3[0] > 1 && inputdims3[1] > 1 && (mxGetNumberOfDimensions(prhs[2]) == 2 || inputdims3[2] == 1))
        {
            if(inputdims[0] == inputdims2[0] && inputdims[0] == inputdims3[0] &&
               inputdims[1] == inputdims2[1] && inputdims[1] == inputdims3[1])
            {
                supposedFormat = NLM_CI_LUMA_CHROMA;
                inputImage.resize(3);
                copyFromArray<IPF_float32>("Y_channel_input", inputImage[0], prhs[0], 0, width, height);
                copyFromArray<IPF_float32>("U_channel_input", inputImage[1], prhs[1], 0, width, height);
                copyFromArray<IPF_float32>("V_channel_input", inputImage[2], prhs[2], 0, width, height);
            }
            else  if(inputdims[0]/2 == inputdims2[0] && inputdims[0]/2 == inputdims3[0] &&
                    inputdims[1]/2 == inputdims2[1] && inputdims[1]/2 == inputdims3[1])
            {
                supposedFormat = NLM_CI_YUV_420;
                inputImage.resize(3);
                copyFromArray<IPF_float32>("Y_channel_input", inputImage[0], prhs[0], 0, width, height);
                copyFromArray<IPF_float32>("U_channel_input", inputImage[1], prhs[1], 0, width/2, height/2);
                copyFromArray<IPF_float32>("V_channel_input", inputImage[2], prhs[2], 0, width/2, height/2);
            }
            else
            {
                mexErrMsgTxt("Input channel data does not seem to fit YUV or YUV420 format");
            }

            if(!checkParam<float>(sigma, "sigma", "sigma", prhs[3]))
                mexErrMsgTxt("could not interpret sigma value");
            paramsoffset = 4;
        }

    }
    else if(mxGetNumberOfDimensions(prhs[0]) == 2 || inputdims[2] == 1)
    {
        supposedFormat = NLM_CI_GREYSCALE;
        inputImage.resize(1);
        copyFromArray<IPF_float32>("Greyscale_input", inputImage[0], prhs[0], 0, width, height);
        if(!checkParam<float>(sigma, "sigma", "sigma", prhs[1]))
            mexErrMsgTxt("could not interpret sigma value");
        paramsoffset = 2;
    }
    else
    {
        supposedFormat = NLM_CI_RGB;
        inputImage.resize(3);
        copyFromArray<IPF_float32>("RGB_input", inputImage[0], prhs[0], 0, width, height);
        copyFromArray<IPF_float32>("RGB_input", inputImage[1], prhs[0], width*height, width, height);
        copyFromArray<IPF_float32>("RGB_input", inputImage[2], prhs[0], 2*width*height, width, height);
        if(!checkParam<float>(sigma, "sigma", "sigma", prhs[1]))
            mexErrMsgTxt("could not interpret sigma value");
        paramsoffset = 2;
    }

    Config config;
    config.Algorithm = NLM_DNA_BM3D_WIENER;
    config.Format = supposedFormat;
    config.BlockSize = 8;
    config.TileSize = 15;
    config.ClusterSize = 32;
    config.NumCandidates = 16;

    std::vector<int> handledParams((nrhs-paramsoffset)/2, 0);
    loadCommonParams(&config, nrhs, prhs, &handledParams[0], paramsoffset);
    checkRemainingParams(nrhs, prhs, &handledParams[0], paramsoffset);

    if(supposedFormat == NLM_CI_GREYSCALE && config.Format != NLM_CI_GREYSCALE)
        mexErrMsgTxt("Only a single input channel is provided, can only use Greyscale.");

    if(supposedFormat == NLM_CI_RGB &&
            (config.Format == NLM_CI_YUV_420 || config.Format == NLM_CI_YUV_420_FAST || config.Format == NLM_CI_YUV_420_FAST_SUBPIXEL) )
        mexErrMsgTxt("Can only use YUV420 when input is already YUV420.");

    if(supposedFormat ==  NLM_CI_RGB && config.Format == NLM_CI_GREYSCALE)
        mexErrMsgTxt("Can only use Greyscale when input is already Greyscale.");

    if(supposedFormat == NLM_CI_YUV_420 &&
            !(config.Format == NLM_CI_YUV_420 || config.Format == NLM_CI_YUV_420_FAST || config.Format == NLM_CI_YUV_420_FAST_SUBPIXEL) )
        mexErrMsgTxt("If input is YUV420 only a YUV420 format can be used.");




    float runtime;
    float memusage;

    Channels inputchannels;
    Channels outputchannels;
    nv::FrameBufferManager& fbm = nv::FrameBufferManager::GetInstance();
    for(int i = 0; i < inputImage.size(); ++i)
    {
        inputchannels.push( gpu::IPF2FrameBuffer(inputImage[i]) );
        outputchannels.push(fbm.Create(inputImage[i].getWidth(), inputImage[i].getHeight(), nv::Type2PixType<float>::PixT, PITCH2D,
                RD_ELEMENT_TYPE)  );
    }

    std::stringstream messages;
    bool success = runDenoiser(config, sigma, inputchannels, outputchannels, memusage, runtime, messages);
    if(messages.str().size() > 1)
        mexWarnMsgTxt(messages.str().c_str());
    if(!success)
        mexErrMsgTxt("Denoising failed - unknown problem");

    std::vector<Image<IPF_float32> > outputImage(inputImage.size());
    for(int i = 0; i < inputImage.size(); ++i)
        outputImage[i] = gpu::FrameBuffer2IPF<IPF_float32>(outputchannels[i]);

	if(nlhs == 1 || nlhs == 2)
	{
	    if(outputImage.size() == 1)
	    {
	        //single channel image
            int ndim = 2;
            int dims[2];
            dims[0] = height;
            dims[1] = width;
            plhs[0] = mxCreateNumericArray(ndim, dims, mxSINGLE_CLASS, mxREAL);
            if (plhs[0] == NULL)
                mexErrMsgTxt("Could not create output mxArray.\n");
            float *R_ptr = (float*)mxGetPr(plhs[0]);
            for(int y = 0; y < height; ++y)
                for(int x = 0; x < width; ++x)
                    R_ptr[x*height + y] =  outputImage[0](x,y);
	    }
	    else if(outputImage[0].getHeight() == outputImage[1].getHeight())
	    {
            //single three channel image
            int ndim = 3;
            int dims[3];
            dims[0] = height;
            dims[1] = width;
            dims[2] = 3;
            plhs[0] = mxCreateNumericArray(ndim, dims, mxSINGLE_CLASS, mxREAL);
            if (plhs[0] == NULL)
                mexErrMsgTxt("Could not create output mxArray.\n");
            float *R_ptr = (float*)mxGetPr(plhs[0]);
            for(int y = 0; y < height; ++y)
                for(int x = 0; x < width; ++x)
                {
                    R_ptr[0 * width * height + x*height + y] =  outputImage[0](x,y);
                    R_ptr[1 * width * height + x*height + y] =  outputImage[1](x,y);
                    R_ptr[2 * width * height + x*height + y] =  outputImage[2](x,y);
                }
	    }
	    else
	    {
	        mexWarnMsgTxt("YUV420 output requested but only one output channel expected, only outputting Luma");
	        //singe Luma image
            int ndim = 2;
            int dims[2];
            dims[0] = height;
            dims[1] = width;
            plhs[0] = mxCreateNumericArray(ndim, dims, mxSINGLE_CLASS, mxREAL);
            if (plhs[0] == NULL)
              mexErrMsgTxt("Could not create output mxArray.\n");
            float *R_ptr = (float*)mxGetPr(plhs[0]);
            for(int y = 0; y < height; ++y)
              for(int x = 0; x < width; ++x)
                  R_ptr[x*height + y] =  outputImage[0](x,y);
	    }
	    if(nlhs == 2)
	    {
	        int tndim = 1;
	        int tdims[1] = {2};
	        plhs[1] = mxCreateNumericArray(tndim, tdims, mxSINGLE_CLASS, mxREAL);
	        *((float*)mxGetPr(plhs[1])) = runtime/1000.0f;
	        ((float*)mxGetPr(plhs[1]))[1] = memusage;
	    }
	}
    else
    {
        // three separate channel output
        int ndim = 2;
        int dims[2];
        for(int i = 0; i < 3; ++i)
        {
            int width = outputImage[i].getWidth();
            int height = outputImage[i].getHeight();
            dims[0] = height;
            dims[1] = width;
            plhs[i] = mxCreateNumericArray(ndim, dims, mxSINGLE_CLASS, mxREAL);
            if (plhs[i] == NULL)
                mexErrMsgTxt("Could not create output mxArray.\n");
            float *R_ptr = (float*) mxGetPr(plhs[i]);
            for (int y = 0; y < height; ++y)
                for (int x = 0; x < width; ++x)
                    R_ptr[x * height + y] = outputImage[i](x, y);
        }
        if(nlhs == 4)
        {
            int tndim = 1;
            int tdims[1] = {2};
            plhs[3] = mxCreateNumericArray(tndim, tdims, mxSINGLE_CLASS, mxREAL);
            *((float*)mxGetPr(plhs[3])) = runtime/1000.0f;
            ((float*)mxGetPr(plhs[3]))[1] = memusage;
        }
    }
    return;
}
