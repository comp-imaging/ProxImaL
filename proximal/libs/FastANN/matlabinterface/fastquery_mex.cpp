
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

    if(nrhs < 2)
    {
        mexPrintf("fastquery_mex:\n");
        mexPrintf("  [datastruct timing] = fastquery_mex(singlechannel_image, indexing_data, [CandidateSize], [PowTwoCandidates])\n");
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

    Image<IPF_float32> inputImage;
    int width, height;

    const int* inputdims = mxGetDimensions(prhs[0]);
    width = inputdims[1];
    height = inputdims[0];

    if (mxGetNumberOfDimensions(prhs[0]) > 2 && inputdims[2] != 1)
    {
        mexPrintf("input image can only have on channel %d\n", inputdims[2]);
        mexErrMsgTxt("Input image not greyscale");
        return;
    }



    // get the data from indexing in
    const char* c_clusteredPixCoords = "clusteredIndices";
    const char* c_clusterOffsets = "clusterOffsets";
    const char* c_clusterSizes = "clusterSizes";
    const char* c_imageDimensions= "dims";
    const char* c_parameters = "parameters";
    const char* c_candidates = "candidates";

    const mxArray* mImageDims = mxGetField(prhs[1], 0, c_imageDimensions);
    if(mImageDims == NULL)
        mexErrMsgTxt("indexing_data is no valid output from fastbuildindex (no dims)");
    int *p_ImageDims = (int*)mxGetPr(mImageDims);
    if(p_ImageDims[0] != height || p_ImageDims[1] != width)
        mexErrMsgTxt("input image size does not match fastbuildindex call");

    const mxArray* mParamaters = mxGetField(prhs[1], 0, c_parameters);
    if (mParamaters == NULL)
        mexErrMsgTxt("indexing_data is no valid output from fastbuildindex (no parameters)");
    int *p_Parameters = (int*) mxGetPr(mParamaters);
    int TileSize = p_Parameters[0],
        BlockSize = p_Parameters[1],
        ClusterSize = p_Parameters[2],
        CandidateSize = std::min<int>(16, ClusterSize);
    bool PowTwoCandidates = true;
    if(nrhs >= 3)
        checkParam<int>(CandidateSize, "CandidateSize", "CandidateSize", prhs[2]);
    if(nrhs >= 4)
        checkParam<bool>(PowTwoCandidates, "PowTwoCandidates", "PowTwoCandidates", prhs[3]);

    copyFromArray<IPF_float32>("InputImage", inputImage, prhs[0], 0, width, height);

    float runtime;
    float memusage;


    nv::FrameBufferManager& fbm = nv::FrameBufferManager::GetInstance();
    nv::FrameBufferHandle in = gpu::IPF2FrameBuffer(inputImage);

    //copy data from indexing in

    std::vector<int2> clusteredPixCoords;
    std::vector<int> clusterOffsets;
    std::vector<int> clusterSizes;

    const mxArray* mClusteredPixCoords = mxGetField(prhs[1], 0, c_clusteredPixCoords);
    if(mClusteredPixCoords == NULL)
        mexErrMsgTxt("indexing_data is no valid output from fastbuildindex (no clusteredIndices)");
    int *pClusteredPixCoords = (int*)mxGetPr(mClusteredPixCoords);
    const int* ClusteredPixCoordsdims = mxGetDimensions(mClusteredPixCoords);
    if(ClusteredPixCoordsdims[0] != 2)
        mexErrMsgTxt("indexing_data is no valid output from fastbuildindex");
    clusteredPixCoords.reserve(ClusteredPixCoordsdims[1]);
    for(int i = 0; i < ClusteredPixCoordsdims[1]; ++i)
        clusteredPixCoords.push_back(make_int2(pClusteredPixCoords[2*i], pClusteredPixCoords[2*i+1]));


    const mxArray* mclusterOffsets = mxGetField(prhs[1], 0, c_clusterOffsets);
    if(mclusterOffsets == NULL)
        mexErrMsgTxt("indexing_data is no valid output from fastbuildindex (no clusterOffsets)");
    int *pClusterOffsets = (int*)mxGetPr(mclusterOffsets);
    const int* clusterOffsetsDims = mxGetDimensions(mclusterOffsets);
    if(clusterOffsetsDims[0] != 1)
        mexErrMsgTxt("indexing_data is no valid output from fastbuildindex");

    const mxArray* mclusterSizes = mxGetField(prhs[1], 0, c_clusterSizes);
    if(mclusterSizes == NULL)
        mexErrMsgTxt("indexing_data is no valid output from fastbuildindex (no clusterSizes)");
    int *pClusterSizes = (int*)mxGetPr(mclusterSizes);
    const int* clusterSizesDims = mxGetDimensions(mclusterSizes);
    if(clusterSizesDims[0] != 1)
        mexErrMsgTxt("indexing_data is no valid output from fastbuildindex");
    if(clusterOffsetsDims[1] != clusterSizesDims[1])
        mexErrMsgTxt("indexing_data is no valid output from fastbuildindex (number of clusterOffsets and clusterSizes does not match)");

    clusterOffsets.reserve(clusterSizesDims[1]);
    clusterSizes.reserve(clusterSizesDims[1]);
    for(int i = 0; i < clusterSizesDims[1]; ++i)
    {
        if(pClusterOffsets[i] < 0 || pClusterOffsets[i]+pClusterSizes[i] > clusteredPixCoords.size())
            mexErrMsgTxt("indexing_data is no valid output from fastbuildindex (clusters point outside of array)");
        clusterOffsets.push_back(pClusterOffsets[i]);
        clusterSizes.push_back(pClusterSizes[i]);
    }

    std::stringstream messages;
    std::vector<int2> candidates;

    bool success = runQuery(candidates, in, clusteredPixCoords,clusterOffsets,clusterSizes, TileSize, BlockSize, ClusterSize, CandidateSize, memusage, runtime, messages, PowTwoCandidates);
    if(messages.str().size() > 1)
        mexWarnMsgTxt(messages.str().c_str());
    if(!success)
        mexErrMsgTxt("Indexing failed - unknown problem");


    const char* pFieldnames[3] =
        {c_candidates,
         c_imageDimensions,
         c_parameters };

    mxArray* outputStruct = mxCreateStructMatrix(1, 1, 3, pFieldnames);
    plhs[0] = outputStruct;

    int candidatesDims[] = {height, width, CandidateSize, 2};
    mxArray* m_candidates = mxCreateNumericArray( 4, candidatesDims, mxINT32_CLASS, mxREAL);
    if(m_candidates == NULL)
        mexErrMsgTxt("Could not create output mxArray.\n");
    int *p_candidates = (int*) mxGetPr(m_candidates);


    std::vector<int2>::iterator it = candidates.begin();
    for(int y = 0; y <  height; ++y)
    for(int x = 0; x <  width; ++x)
    for(int c = 0; c <  CandidateSize && it != candidates.end(); ++c, ++it)
    {
        int2 coords = *it;
//        p_candidates[(x*height + y)*2*CandidateSize + 2*c] =  coords.x;
//        p_candidates[(x*height + y)*2*CandidateSize + 2*c + 1] =  coords.y;

        p_candidates[x*height + y + c * width * height] = coords.x;
        p_candidates[x*height + y + (c  + CandidateSize) * width * height] = coords.y;
    }


    mxArray* m_imageDimensions = mxCreateNumericMatrix( 2, 1, mxINT32_CLASS, mxREAL);
    if(m_imageDimensions == NULL)
        mexErrMsgTxt("Could not create output mxArray.\n");
    int *p_imageDimensions = (int*)mxGetPr(m_imageDimensions);
    p_imageDimensions[0] = height;
    p_imageDimensions[1] = width;

    mxArray* m_parameters = mxCreateNumericMatrix( 4, 1, mxINT32_CLASS, mxREAL);
    if(m_parameters == NULL)
        mexErrMsgTxt("Could not create output mxArray.\n");
    int *p_parameters = (int*)mxGetPr(m_parameters);
    p_parameters[0] = TileSize;
    p_parameters[1] = BlockSize;
    p_parameters[2] = ClusterSize;
    p_parameters[3] = CandidateSize;

    // set the data to the array
    mxSetFieldByNumber(outputStruct, 0, 0,  m_candidates);
    mxSetFieldByNumber(outputStruct, 0, 1,  m_imageDimensions);
    mxSetFieldByNumber(outputStruct, 0, 2,  m_parameters);

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
