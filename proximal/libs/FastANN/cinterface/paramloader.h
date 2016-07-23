
#include <string>
#include <sstream>
#include <iostream>
#include <ip/Image.h>

#include <cmath>
#include <math.h>
#include "gpuinterface.h"

template<ImagePixelFormat TFormat>
bool copyFromArray(const std::string& name, Image<TFormat>& img, float *p, int offset, int w, int h, float scale = 1.f)
{
    img =  Image<TFormat>(w,h);
    for(int y = 0; y < h; ++y)
        for(int x = 0; x < w; ++x)
            img(x,y) = p[x*h + y + offset]*scale;

    return true;
}

template<class T>
bool checkParam(T& toset, const std::string& name, const float param, bool verbose)
{

    toset = (T)param;
    if(verbose)
        std::cout << "Setting parameter " << name << " to " <<  toset << std::endl;
    return true;
}

void loadCommonParams(Config* config, int numParams, float *params, bool verbose)
{
    for(int i = 0; i < numParams; i++)
    {

        if( i == 0 ) //"Algorithm" (0 - SlidingDCT, 1 - NlmAverage, 2 - NlmWeightedAverage, 3 - BM3D, 4 - BM3D Wiener)
        {
            int denoisername = (int)(params[i]);
            switch (denoisername) {

              case 0: //"SlidingDCT"
                config->Algorithm = NLM_DNA_DCT_SOFTTHRESHOLD;
                if(verbose)
                    std::cout << "Using SlidingDCT" << std::endl;
                break;

              case 1: //"NlmAverage"
                config->Algorithm = NLM_DNA_AVERAGE;
                if(verbose)
                    std::cout << "Using Simple Average NLM" << std::endl;
                break;

              case 2: //"NlmWeightedAverage"
                config->Algorithm = NLM_DNA_WEIGHTED_AVERAGE;
                if(verbose)
                    std::cout << "Using Weighted Average NLM" << std::endl;
                break;

              case 3: //"BM3D"
                config->Algorithm = NLM_DNA_BM3D;
                if(verbose)
                    std::cout << "Using BM3D hard thresholding" << std::endl;
                break;

              case 4: //"BM3D Wiener"
                config->Algorithm = NLM_DNA_BM3D_WIENER;
                if(verbose)
                    std::cout << "Using BM3D Wiener filtering" << std::endl;
                break;

              default:
                std::cerr << "Unsupported Denoising method." << std::endl;
            }

        }
        else if( i == 1) //"Format" (0 - RGBNoConvert, 1 - RGB, 2 - YUV420, 3 - Greyscale, 4 - LumaChroma)
        {
            int formatname =  (int)(params[i]);
            switch (formatname) {

              case 0: //"RGBNoConvert"
                config->Format = NLM_CI_RGB;
                if(verbose)
                    std::cout << "Running in pure RGB mode" << std::endl;
                break;

              case 1: //"RGB"
                config->Format = NLM_CI_RGB_CONVERT_OPP;
                if(verbose)
                    std::cout << "Running in RGB (Opp space) mode" << std::endl;
                break;

              case 2: //"YUV420"
                config->Format = NLM_CI_YUV_420;
                if(verbose)
                    std::cout << "Running in YUV420 mode" << std::endl;
                break;

              case 3: //"Greyscale"
                config->Format = NLM_CI_GREYSCALE;
                if(verbose)
                    std::cout << "Running in greyscale mode" << std::endl;
                break;

              case 4: //"LumaChroma"
                config->Format = NLM_CI_LUMA_CHROMA;
                if(verbose)
                    std::cout << "Running in luma chroma mode" << std::endl;
                break;

              default:
                std::cerr << "Unsupported Format." << std::endl;
            }
        
        }
        else if(i == 2) //"BlockSize"
            checkParam<int>(config->BlockSize,"BlockSize",params[i], verbose);
        else if(i == 3)  //"TileSize"
            checkParam<int>(config->TileSize,"TileSize",params[i], verbose);
        else if(i == 4)  //"NumCandidates"
            checkParam<int>(config->NumCandidates,"NumCandidates",params[i], verbose);
        else if(i == 5)  //"ClusterSize"
            checkParam<int>(config->ClusterSize,"ClusterSize",params[i], verbose);

    }
}

void printCommonParams()
{
    std::cout << " 'Algorithm' denoising algorithm to be used, possible settings: 'NlmWeightedAverage', 'BM3D', 'BM3DWiener', 'NlmAverage', 'SlidingDCT'" << std::endl;
    std::cout << " 'Format' data format to run the algorithm on, possible settings: 'Greyscale', 'RGB', 'RGBNoConvert', 'LumaChroma', 'YUV420'" << std::endl;
    std::cout << " 'BlockSize' (int) dimension of a single patch (BlockSizexBlocksize) possible values 4, 8, 16" << std::endl;
    std::cout << " 'TileSize' (int) dimension of one tile (11-21)" << std::endl;
    std::cout << " 'NumCandidates' (int) number of candidates to be considered (8, 16)" << std::endl;
    std::cout << " 'ClusterSize' (int) maximum number of elements to end up in a cluster (16, 32, 64)" << std::endl;
}