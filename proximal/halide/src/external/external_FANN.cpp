////////////////////////////////////////////////////////////////////////////////
//External call for FANN from FASTANN Library
////////////////////////////////////////////////////////////////////////////////

#include <map>
#include <vector>
#include <complex>
#include <chrono>
#include <iostream>
#include <thread>

#include "static_image.h"
#include <cinterface/FANN_denoise.h>
#include "math.h"

using namespace std;

extern "C" int FANN_extern(buffer_t * in, buffer_t * params, float sigma, int verbose, int w, int h, int ch, int numparams, buffer_t * out)
{
    //Bounds
    if (in->host == NULL) {
        in->min[0] = 0;
        in->min[1] = 0;
        in->min[2] = 0;
        in->extent[0] = w;
        in->extent[1] = h;
        in->extent[2] = ch;

        params->min[0] = 0;
        params->extent[0] = numparams;
        
        return 0;
    }

    //Assert type
    assert(in->elem_size == sizeof(float));
    assert(out->elem_size == sizeof(float));

    // Check the inputs are as large as we expected.
    assert(in->min[0] == 0 && in->min[1] == 0 && in->min[2] == 0);

    // Check the strides are what we want.
    assert(in->stride[0] == 1 && in->stride[1] == w && in->stride[2] == w * h && 
          out->stride[0] == 1 && out->stride[1] == w && out->stride[2] == w * h);

    //Get buffers
    float *in_buf = (float *)(in->host);
    float *out_buf = (float *)(out->host);
    float *param_buf = (float *)(params->host);
   
    //###### Do the denoising #######
    if(verbose)
        cout << "Running FANN denoising with sigma: [sigma =" << sigma << "] !"<< endl;

    float* stats = new float[2];
    run_FANN_denoise(in_buf, w, h, ch, sigma, param_buf, numparams, verbose, out_buf, stats);

    if(verbose)
        cout << "FANN denosing took " << roundf(1000 * stats[0]) << 
                " ms and used " << stats[1]/1024/1024 << "MB of memory." << endl;

    delete[] stats;
	return 0;
}
