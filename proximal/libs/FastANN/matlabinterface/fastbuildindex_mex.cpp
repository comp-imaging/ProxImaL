
//Includes
#include "mex.h"
#include "matrix.h"
#include "gpuinterface_separate.h"
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

    if(nrhs < 1)
    {
        mexPrintf("fastbuidindex_mex:\n");
        mexPrintf("  [datastruct timing] = fastbuidindex_mex(singlechannel_image, [TileSize], [BlockSize], [ClusterSize])\n");
        mexErrMsgTxt("Insufficient number of inputs");
        return;
    }
    if(nlhs < 1)
    {
        mexErrMsgTxt("Insufficient number of outputs");
    }
    if(nlhs > 2)
    {
        mexErrMsgTxt("Too many output parameters");
    }

    int TileSize = 15,
        BlockSize = 8,
        ClusterSize = 32;
    if (nrhs >= 2)
        checkParam<int>(TileSize, "TileSize", "TileSize", prhs[1]);
    if (nrhs >= 3)
        checkParam<int>(BlockSize, "BlockSize", "BlockSize", prhs[2]);
    if (nrhs >= 4)
        checkParam<int>(ClusterSize, "ClusterSize", "ClusterSize", prhs[3]);


    const int* inputdims = mxGetDimensions(prhs[0]);


    Image<IPF_float32>  inputImage;
    int width, height;


    if(mxGetNumberOfDimensions(prhs[0]) > 2 && inputdims[2] != 1)
    {
        mexPrintf("input image can only have on channel %d\n", inputdims[2]);
        mexErrMsgTxt("Input image not greyscale");
        return;
    }
    width = inputdims[1];
    height = inputdims[0];

    copyFromArray<IPF_float32>("InputImage", inputImage, prhs[0], 0, width, height);

    float runtime;
    float memusage;


    nv::FrameBufferManager& fbm = nv::FrameBufferManager::GetInstance();
    nv::FrameBufferHandle in = gpu::IPF2FrameBuffer(inputImage);

    std::vector<int2> clusteredPixCoords;
    std::vector<int> clusterOffsets;
    std::vector<int> clusterSizes;

    std::stringstream messages;

    bool success = runIndexing(clusteredPixCoords,clusterOffsets,clusterSizes, in, TileSize, BlockSize, ClusterSize, memusage, runtime, messages);
    if(messages.str().size() > 1)
        mexWarnMsgTxt(messages.str().c_str());
    if(!success)
        mexErrMsgTxt("Indexing failed - unknown problem");

    const char* c_clusteredPixCoords = "clusteredIndices";
    const char* c_clusterOffsets = "clusterOffsets";
    const char* c_clusterSizes = "clusterSizes";
    const char* c_imageDimensions= "dims";
    const char* c_parameters = "parameters";
    const char* pFieldnames[5] =
        {c_clusteredPixCoords,
         c_clusterOffsets,
         c_clusterSizes,
         c_imageDimensions,
         c_parameters };

    mxArray* outputStruct = mxCreateStructMatrix(1, 1, 5, pFieldnames);
    plhs[0] = outputStruct;

    int clusteredPixCoordsDims[] = {2, clusteredPixCoords.size()};
    mxArray* m_clusteredPixCoords = mxCreateNumericArray( 2, clusteredPixCoordsDims, mxINT32_CLASS, mxREAL);
    if(m_clusteredPixCoords == NULL)
        mexErrMsgTxt("Could not create output mxArray.\n");
    int *p_clusteredPixCoords = (int*)mxGetPr(m_clusteredPixCoords);
    for(int i = 0; i <  clusteredPixCoords.size(); ++i)
    {
        p_clusteredPixCoords[2*i] = clusteredPixCoords[i].x;
        p_clusteredPixCoords[2*i + 1] = clusteredPixCoords[i].y;
    }

    int clusterOffsetsDims[] = {1, clusterOffsets.size()};
    mxArray* m_clusterOffsets = mxCreateNumericArray( 2, clusterOffsetsDims, mxINT32_CLASS, mxREAL);
    if(m_clusterOffsets == NULL)
        mexErrMsgTxt("Could not create output mxArray.\n");
    int *p_clusterOffsets = (int*)mxGetPr(m_clusterOffsets);
    for(int i = 0; i <  clusterOffsets.size(); ++i)
        p_clusterOffsets[i] = clusterOffsets[i];

    int clusterSizesDims[] = {1, clusterSizes.size()};
    mxArray* m_clusterSizes = mxCreateNumericArray( 2, clusterSizesDims, mxINT32_CLASS, mxREAL);
    if(m_clusterSizes == NULL)
        mexErrMsgTxt("Could not create output mxArray.\n");
    int *p_clusterSizes = (int*)mxGetPr(m_clusterSizes);
    for(int i = 0; i <  clusterSizes.size(); ++i)
        p_clusterSizes[i] = clusterSizes[i];

    mxArray* m_imageDimensions = mxCreateNumericMatrix( 2, 1, mxINT32_CLASS, mxREAL);
    if(m_imageDimensions == NULL)
        mexErrMsgTxt("Could not create output mxArray.\n");
    int *p_imageDimensions = (int*)mxGetPr(m_imageDimensions);
    p_imageDimensions[0] = height;
    p_imageDimensions[1] = width;

    mxArray* m_parameters = mxCreateNumericMatrix( 3, 1, mxINT32_CLASS, mxREAL);
    if(m_parameters == NULL)
        mexErrMsgTxt("Could not create output mxArray.\n");
    int *p_parameters = (int*)mxGetPr(m_parameters);
    p_parameters[0] = TileSize;
    p_parameters[1] = BlockSize;
    p_parameters[2] = ClusterSize;

    // set the data to the array
    mxSetFieldByNumber(outputStruct, 0, 0,  m_clusteredPixCoords);
    mxSetFieldByNumber(outputStruct, 0, 1,  m_clusterOffsets);
    mxSetFieldByNumber(outputStruct, 0, 2,  m_clusterSizes);
    mxSetFieldByNumber(outputStruct, 0, 3,  m_imageDimensions);
    mxSetFieldByNumber(outputStruct, 0, 4,  m_parameters);

    if(nlhs == 2)
    {
        int tndim = 1;
        int tdims[1] = {2};
        plhs[1] = mxCreateNumericArray(tndim, tdims, mxSINGLE_CLASS, mxREAL);
        *((float*)mxGetPr(plhs[1])) = runtime/1000.0f;
        ((float*)mxGetPr(plhs[1]))[1] = memusage;
    }

    return;
}
