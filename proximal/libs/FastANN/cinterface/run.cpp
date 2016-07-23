#include <iostream>
#include <stdio.h>
#include "math.h"

#include "FANN_denoise.h"

using namespace std;

int main(int argc, char* argv[])
{
	//Patch definitions
	int w = 256;
	int h = 256;

	int verbose = 1;
	
	//Load blurred image
	float* input = new float[w * h * 3];
	float* output = new float[w * h * 3];
	
	//Fill
	for(int ch = 0; ch < 3; ch++ )
	{
		for(int y = 0; y < h; y++ )
		{
		  for(int x = 0; x < w; x++ )
		  {
		    input[y * w + x + ch * w * h] = ((float)(y * w + x))/((float)(w * h));
		  }
		}
	}
	
	//Debug
	if(verbose)
		printf("Size: width %d, height %d\n", w, h);

	//Load blurred image
	float* params = new float[6];
	float* stats = new float[2];
	int numparams = 2;
	
	int alg = 4;
	int form = 1;
	int blocksize = 8;
	int tilesize = 15;
	int numcandidates = 16;
	int clustersize = 32;

	params[0] = alg;
	params[1] = form;
	params[2] = blocksize;
	params[3] = tilesize;
	params[4] = numcandidates;
	params[5] = clustersize;

	int sigma_denoise = 0.05f;

	run_FANN_denoise(input, w, h, 3, 
                      sigma_denoise, params, numparams, verbose, 
                      output, stats);


    cout << "FANN denosing took " << roundf(1000 * stats[0]) << 
            " ms and used " << stats[1]/1024/1024 << "MB of memory." << endl;



	//Release interface
	delete[] params;
	delete[] input;
	delete[] output;
	delete[] stats;
	return 0;
}
