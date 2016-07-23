
//Includes
#include "mex.h"
#include "matrix.h"
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

float L2(const Image<IPF_float32>& img, int sx, int sy, int tx, int ty, int BlockSize)
{
    float sum = 0;
    for(int iy = 0; iy < BlockSize; ++iy)
    for(int ix = 0; ix < BlockSize; ++ix)
    {
        float diff = img(sx + ix, sy+iy) - img(tx + ix, ty+iy);
        sum += diff*diff;
    }
    return sum;
}

void mexFunction(int nlhs, mxArray *plhs[], /* Output variables */
         int nrhs, const mxArray *prhs[]) /* Input variables */
{

    if(nrhs < 1)
    {
        mexPrintf("query_gt_mex:\n");
        mexPrintf("  [candidates] = query_gt_mex(singlechannel_image, [BlockSize], [CandidateSize], [PowTwoCandidates], [SearchRadius])\n");
        mexErrMsgTxt("Insufficient number of inputs");
        return;
    }
    if(nlhs < 1)
    {
        mexErrMsgTxt("Insufficient number of outputs");
    }
    if(nlhs > 1)
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



    int BlockSize = 8,
        CandidateSize = 16,
        SearchRadius = std::max(width,height);
    bool PowTwoCandidates = true;
    if(nrhs >= 2)
        checkParam<int>(BlockSize, "BlockSize", "BlockSize", prhs[1]);
    if(nrhs >= 3)
        checkParam<int>(CandidateSize, "CandidateSize", "CandidateSize", prhs[2]);
    if(nrhs >= 4)
        checkParam<bool>(PowTwoCandidates, "PowTwoCandidates", "PowTwoCandidates", prhs[3]);
    if(nrhs >= 5)
        checkParam<int>(SearchRadius, "SearchRadius", "SearchRadius", prhs[4]);

    copyFromArray<IPF_float32>("InputImage", inputImage, prhs[0], 0, width, height);

    std::vector<int2> candidates(width*height*CandidateSize);

    for(int y = 0; y < height-BlockSize+1; ++y)
    {
        mexPrintf("starting search for row %04d/%04d", y+1, height-BlockSize+1);
        mexEvalString("drawnow;");
        for(int x = 0; x < width-BlockSize+1; ++x)
        {

            std::vector<float> dist(CandidateSize, 100000000.0f);
            std::vector<int2> tcandidates(CandidateSize, make_int2(0,0));
            //search
            for(int sy = std::max(0, y-SearchRadius); sy < std::min(height-BlockSize+1,y+SearchRadius); ++sy)
                for(int sx = std::max(0, x-SearchRadius); sx < std::min(width-BlockSize+1,x+SearchRadius); ++sx)
                {
                    // comp dist
                    float n = L2(inputImage, sx, sy, x, y, BlockSize);
                    //mexPrintf("%d %d %d %d distance is %f\n", sx, sy, x, y,n);
                    if(dist.back() > n)
                    {
                        int i = CandidateSize-2;
                        for(; i >= 0 && dist[i] > n; --i)
                        {
                            dist[i+1] = dist[i];
                            tcandidates[i+1] = tcandidates[i];
                        }
                        //mexPrintf("inserting @ %d\n", i+1);
                        dist[i+1] = n;
                        tcandidates[i+1] = make_int2(sx+1, sy+1);
                    }
                }
            for(int i = 0; i < CandidateSize; ++i)
                candidates[(y*width + x)*CandidateSize + i] = tcandidates[i];
        }
        mexPrintf("\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b");
    }


    int candidatesDims[] = {height, width, CandidateSize, 2};
    mxArray* m_candidates = mxCreateNumericArray( 4, candidatesDims, mxINT32_CLASS, mxREAL);
    if(m_candidates == NULL)
        mexErrMsgTxt("Could not create output mxArray.\n");
    plhs[0] = m_candidates;
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

    return;
}
