
#include "mex.h"
#include "matrix.h"
#include <string>
#include <sstream>
#include <ip/Image.h>

#include <cmath>
#include <math.h>
#include "gpuinterface.h"



/**
 * convert param to string
 */
std::string getString(const mxArray *param)
{
    if(mxGetClassID(param) != mxCHAR_CLASS)
        return std::string();

    const int * stringdims = mxGetDimensions(param);
    const mxChar * thestring = (const mxChar *)mxGetData(param);
    size_t l = stringdims[1];

    std::string str(l,'\0');
    for(int i = 0; i < l; ++i)
        str[i] = thestring[i];
    return str;
}

template<ImagePixelFormat TFormat>
bool copyFromArray(const std::string& name, Image<TFormat>& img, const mxArray *param, int offset, int w, int h, float scale = 1.0f)
{
    img =  Image<TFormat>(w,h);
    const void* value = (const void*)mxGetData(param);
    switch(mxGetClassID(param))
    {
    case mxLOGICAL_CLASS:
    {
        const mxLogical* p = reinterpret_cast<const mxLogical*>(value) + offset;
        for(int y = 0; y < h; ++y)
            for(int x = 0; x < w; ++x)
                img(x,y) = p[x*h + y]*scale;
        break;
    }
    case mxCHAR_CLASS:
    {
        const mxChar* p = reinterpret_cast<const mxChar*>(value) + offset;
        for(int y = 0; y < h; ++y)
            for(int x = 0; x < w; ++x)
                img(x,y) = p[x*h + y]*scale;
        break;
    }
    case mxDOUBLE_CLASS:
    {
        const double* p = reinterpret_cast<const double*>(value)+ offset;
        for(int y = 0; y < h; ++y)
            for(int x = 0; x < w; ++x)
                img(x,y) = p[x*h + y]*scale;
        break;
    }
    case mxINT8_CLASS:
    {
        const char* p = reinterpret_cast<const char*>(value)+ offset;
        for(int y = 0; y < h; ++y)
            for(int x = 0; x < w; ++x)
                img(x,y) = p[x*h + y]*scale;
        break;
    }
    case mxUINT8_CLASS:
    {
        const unsigned char* p = reinterpret_cast<const unsigned char*>(value)+ offset;
        for(int y = 0; y < h; ++y)
            for(int x = 0; x < w; ++x)
                img(x,y) = p[x*h + y]*scale;
        break;
    }
    case mxINT16_CLASS:
    {
        const short* p = reinterpret_cast<const short*>(value)+ offset;
        for(int y = 0; y < h; ++y)
            for(int x = 0; x < w; ++x)
                img(x,y) = p[x*h + y]*scale;
        break;
    }
    case mxUINT16_CLASS:
    {
        const unsigned short* p = reinterpret_cast<const unsigned short*>(value)+ offset;
        for(int y = 0; y < h; ++y)
            for(int x = 0; x < w; ++x)
                img(x,y) = p[x*h + y]*scale;
        break;
    }
    case mxINT32_CLASS:
    {
        const int* p = reinterpret_cast<const int*>(value)+ offset;
        for(int y = 0; y < h; ++y)
            for(int x = 0; x < w; ++x)
                img(x,y) = p[x*h + y]*scale;
        break;
    }
    case mxUINT32_CLASS:
    {
        const unsigned int* p = reinterpret_cast<const unsigned int*>(value)+ offset;
        for(int y = 0; y < h; ++y)
            for(int x = 0; x < w; ++x)
                img(x,y) = p[x*h + y]*scale;
        break;
    }
    case mxINT64_CLASS:
    {
        const long long* p = reinterpret_cast<const long long*>(value)+ offset;
        for(int y = 0; y < h; ++y)
            for(int x = 0; x < w; ++x)
                img(x,y) = p[x*h + y]*scale;
        break;
    }
    case mxUINT64_CLASS:
    {
        const unsigned long long* p = reinterpret_cast<const unsigned long long*>(value)+ offset;
        for(int y = 0; y < h; ++y)
            for(int x = 0; x < w; ++x)
                img(x,y) = p[x*h + y]*scale;
        break;
    }
    case mxSINGLE_CLASS:
    {
        const float* p = reinterpret_cast<const float*>(value)+ offset;
        for(int y = 0; y < h; ++y)
            for(int x = 0; x < w; ++x)
                img(x,y) = p[x*h + y]*scale;
        break;
    }
    default:
        mexPrintf("WARNING: could not interpret data type for '%s'\n",name.c_str());
        return false;
    }
    return true;
}

template<ImagePixelFormat TFormat>
bool copyMatrix(Image<TFormat>& img, const std::string& name, const mxArray *param)
{
    const int * dims = mxGetDimensions(param);
    return copyFromArray<TFormat>(name, img, param, 0, dims[1], dims[0]);
}

template<class T, int w, int h>
bool copyFixedMatrix(T out[h][w], const std::string& name, const mxArray *param)
{
    const int * dims = mxGetDimensions(param);
    if(dims[0] != h || dims[1] != w)
    {
        mexPrintf("WARNING: dimensions do not match for '%s'. expected: %d %d, are: %d %d\n",name.c_str(),w,h,  dims[1],  dims[0]);
        return false;
    }
    const void* value = (const void*)mxGetData(param);
    switch(mxGetClassID(param))
    {
    case mxLOGICAL_CLASS:
    {
        const mxLogical* p = reinterpret_cast<const mxLogical*>(value);
        for(int y = 0; y < h; ++y)
            for(int x = 0; x < w; ++x)
                out[y][x] = p[x*h + y];
        break;
    }
    case mxCHAR_CLASS:
    {
        const mxChar* p = reinterpret_cast<const mxChar*>(value);
        for(int y = 0; y < h; ++y)
            for(int x = 0; x < w; ++x)
                out[y][x] = p[x*h + y];
        break;
    }
    case mxDOUBLE_CLASS:
    {
        const double* p = reinterpret_cast<const double*>(value);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                out[y][x] = p[x*h + y];
        break;
    }
    case mxINT8_CLASS:
    {
        const char* p = reinterpret_cast<const char*>(value);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                out[y][x] = p[x*h + y];
        break;
    }
    case mxUINT8_CLASS:
    {
        const unsigned char* p = reinterpret_cast<const unsigned char*>(value);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                out[y][x] = p[x*h + y];
        break;
    }
    case mxINT16_CLASS:
    {
        const short* p = reinterpret_cast<const short*>(value);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                out[y][x] = p[x*h + y];
        break;
    }
    case mxUINT16_CLASS:
    {
        const unsigned short* p = reinterpret_cast<const unsigned short*>(value);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                out[y][x] = p[x*h + y];
        break;
    }
    case mxINT32_CLASS:
    {
        const int* p = reinterpret_cast<const int*>(value);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                out[y][x] = p[x*h + y];
        break;
    }
    case mxUINT32_CLASS:
    {
        const unsigned int* p = reinterpret_cast<const unsigned int*>(value);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                out[y][x] = p[x*h + y];
        break;
    }
    case mxINT64_CLASS:
    {
        const long long* p = reinterpret_cast<const long long*>(value);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                out[y][x] = p[x*h + y];
        break;
    }
    case mxUINT64_CLASS:
    {
        const unsigned long long* p = reinterpret_cast<const unsigned long long*>(value);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                out[y][x] = p[x*h + y];
        break;
    }
    case mxSINGLE_CLASS:
    {
        const float* p = reinterpret_cast<const float*>(value);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                out[y][x] = p[x*h + y];
        break;
    }
    default:
        mexPrintf("WARNING: could not interpret data type for '%s'\n",name.c_str());
        return false;
    }

    mexPrintf("Setting '%s' to:\n",name.c_str());
    for (int y = 0; y < h; ++y)
    {
        mexPrintf("  ");
        for (int x = 0; x < w; ++x)
            mexPrintf(" %f",(float) out[y][x]);
        mexPrintf("\n");
    }
    return true;
}

template<class T>
bool checkParam(T& toset, const std::string& name, const std::string& identifier, const mxArray *param)
{
    if(name.compare(identifier) == 0)
    {
        const void* value = (const void*)mxGetData(param);
        T t = -1;
        bool ok = true;
        switch(mxGetClassID(param))
        {
        case mxLOGICAL_CLASS:
            t = *reinterpret_cast<const mxLogical*>(value) ? 1 : 0;
            break;
        case mxCHAR_CLASS:
            t = *reinterpret_cast<const mxChar*>(value);
            break;
        case mxDOUBLE_CLASS:
            t = *reinterpret_cast<const double*>(value);
            break;
        case mxINT8_CLASS:
            t = *reinterpret_cast<const char*>(value);
            break;
        case mxUINT8_CLASS:
            t = *reinterpret_cast<const unsigned char*>(value);
            break;
        case mxINT16_CLASS:
            t = *reinterpret_cast<const short*>(value);
            break;
        case mxUINT16_CLASS:
            t = *reinterpret_cast<const unsigned short*>(value);
            break;
        case mxINT32_CLASS:
            t = *reinterpret_cast<const int*>(value);
            break;
        case mxUINT32_CLASS:
            t = *reinterpret_cast<const unsigned int*>(value);
            break;
        case mxINT64_CLASS:
            t = *reinterpret_cast<const long long*>(value);
            break;
        case mxUINT64_CLASS:
            t = *reinterpret_cast<const unsigned long long*>(value);
            break;
        case mxSINGLE_CLASS:
            t = *reinterpret_cast<const float*>(value);
            break;
        default:
            mexPrintf("WARNING: could not interpret setting for '%s'\n",name.c_str());
            ok = false;
            break;
        }

        if(ok)
        {
            toset = t;
            std::stringstream sstr;
            sstr << t;
            mexPrintf("Setting parameter '%s' to %s\n",name.c_str(),sstr.str().c_str());
        }
        return true;
    }
    return false;
}
void loadCommonParams(Config* config, int numParams, const mxArray *params[], int * handled_param, int offset = 0)
{
    for(int i = offset; i < numParams; i+=2)
    {
        if(handled_param[(i-offset)/2] == 1)
            continue;
        bool handled = false;
        std::string tparamname = getString(params[i]);
        if(checkParam<int>(config->BlockSize,"BlockSize",tparamname,params[i+1]) ) handled = true;
        else if(checkParam<int>(config->TileSize,"TileSize",tparamname,params[i+1]) ) handled = true;
        else if(checkParam<int>(config->NumCandidates,"NumCandidates",tparamname,params[i+1]) ) handled = true;
        else if(checkParam<int>(config->ClusterSize,"ClusterSize",tparamname,params[i+1]) ) handled = true;

        else if(tparamname.compare("Algorithm")==0)
        {
            std::string denoisername =  getString(params[i+1]);
            if(denoisername.compare("SlidingDCT")==0)
                mexPrintf("Using SlidingDCT\n"),
                config->Algorithm = NLM_DNA_DCT_SOFTTHRESHOLD;
            else if (denoisername.compare("NlmAverage") == 0)
                mexPrintf("Using Simple Average NLM\n"),
                config->Algorithm = NLM_DNA_AVERAGE;
            else if (denoisername.compare("NlmWeightedAverage") == 0)
                mexPrintf("Using Weighted Average NLM\n"),
                config->Algorithm = NLM_DNA_WEIGHTED_AVERAGE;
            else if (denoisername.compare("BM3D") == 0)
                mexPrintf("Using BM3D hard thresholding\n"),
                config->Algorithm = NLM_DNA_BM3D;
            else if (denoisername.compare("BM3DWiener") == 0)
                mexPrintf("Using BM3D Wiener filtering\n"),
                config->Algorithm = NLM_DNA_BM3D_WIENER;
            else
                mexPrintf("WARNING: unknown denoiser: '%s'\n"
                       "supported values are 'SlidingDCT', 'NlmAverage', 'NlmFastWeightedAverage', 'NlmWeightedAverage', 'BM3D', 'BM3DWiener'\n", denoisername.c_str());
            handled = true;
        }

        else if(tparamname.compare("Format")==0)
        {
            std::string channelinfoname =  getString(params[i+1]);
            if(channelinfoname.compare("RGBNoConvert")==0)
                mexPrintf("Running in pure RGB mode\n"),
                config->Format = NLM_CI_RGB;
            else if(channelinfoname.compare("RGB")==0)
                mexPrintf("Running in RGB (Opp space) mode\n"),
                config->Format = NLM_CI_RGB_CONVERT_OPP;
            else if (channelinfoname.compare("YUV420") == 0)
                mexPrintf("Running in YUV420 mode\n"),
                config->Format = NLM_CI_YUV_420;
            else if (channelinfoname.compare("YUV420Fast") == 0)
                mexPrintf("Running in YUV420 fast mode\n"),
                config->Format = NLM_CI_YUV_420_FAST;
            else if (channelinfoname.compare("YUV420FastSubpixel") == 0)
                mexPrintf("Running in YUV420 fast subpixel mode\n"),
                config->Format = NLM_CI_YUV_420_FAST_SUBPIXEL;
            else if (channelinfoname.compare("Greyscale") == 0)
                mexPrintf("Running in greyscale mode\n"),
                config->Format = NLM_CI_GREYSCALE;
            else if (channelinfoname.compare("LumaChroma") == 0)
                mexPrintf("Running in luma chroma mode\n"),
                config->Format = NLM_CI_LUMA_CHROMA;
            else
                mexPrintf("WARNING: unknown channelInfo: '%s'\n"
                       "supported values are 'RGB', 'YUV420'\n",channelinfoname.c_str());
            handled = true;
        }
        handled_param[(i-offset)/2] = handled ? 1 : 0;
    }
}

void printCommonParams()
{
    mexPrintf("    'BlockSize' (int) dimension of a single patch (BlockSizexBlocksize) possible values 4, 8, 16\n");
    mexPrintf("    'TileSize' (int) dimension of one tile (11-21)\n");
    mexPrintf("    'ClusterSize' (int) maximum number of elements to end up in a cluster (16, 32, 64)\n");
    mexPrintf("    'NumCandidates' (int) number of candidates to be considered (8, 16)\n");

    mexPrintf("    'Algorithm' denoising algorithm to be used, possible settings:\n      'NlmWeightedAverage', 'BM3D', 'BM3DWiener', 'NlmAverage', 'SlidingDCT'\n");
    mexPrintf("    'Format' data format to run the algorithm on, possible settings:\n      'Greyscale', 'RGB', 'RGBNoConvert', 'LumaChroma', 'YUV420', 'YUV420Fast', 'YUV420FastSubpixel' \n");
}

void checkRemainingParams(int numParams, const mxArray *params[], int * handled_param, int offset = 0)
{;
    for(int i = offset; i < numParams; i+=2)
    {
        if(handled_param[i/2] == 0)
        {
            std::string tparamname = getString(params[i]);
            mexPrintf("WARNING: unknown parameter: '%s'\n",tparamname.c_str());
        }
    }
}
