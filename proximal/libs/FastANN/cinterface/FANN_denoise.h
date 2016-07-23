
#ifndef FANN_DENOISE_H_
#define FANN_DENOISE_H_

// Params - [Algorithm, Format, (BlockSize, TileSize, NumCandidates, ClusterSize)];
// Algorithm" (0 - SlidingDCT, 1 - NlmAverage, 2 - NlmWeightedAverage, 3 - BM3D, 4 - BM3D Wiener)
// "Format" (0 - RGBNoConvert, 1 - RGB, 2 - YUV420, 3 - Greyscale, 4 - LumaChroma)
// Arrays are all heigh x width

void run_FANN_denoise(float* input, int width, int height, int channels, 
                      float sigma, float* params, int numParams, bool verbose, 
                      float* output, float* stats = nullptr);


#endif //FANN_DENOISE_H_
