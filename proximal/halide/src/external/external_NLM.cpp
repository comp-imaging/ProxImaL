////////////////////////////////////////////////////////////////////////////////
//External call for NLM from OpenCV (cuda)
////////////////////////////////////////////////////////////////////////////////

#include <map>
#include <vector>
#include <complex>
#include <chrono>
#include <iostream>
#include <thread>

#include "static_image.h"

#include <opencv2/cudaarithm.hpp>
#include <opencv2/photo/cuda.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;
using namespace cv::cuda;

extern "C" int NLM_extern(buffer_t * in, buffer_t * params, float sigma, int w, int h, int ch, buffer_t * out)
{
    //Bounds
    if (in->host == NULL) {
        in->min[0] = 0;
        in->min[1] = 0;
        in->min[2] = 0;
        in->extent[0] = ch;
        in->extent[1] = w;
        in->extent[2] = h;
        
        params->min[0] = 0;
        params->extent[0] = 4;
        
        return 0;
    }

    //Assert type
    assert(in->elem_size == sizeof(float));
    assert(out->elem_size == sizeof(float));

    // Check the inputs are as large as we expected.
    assert(in->min[0] == 0 && in->min[1] == 0 && in->min[2] == 0);

    /*
    //Debug
    printf("Strides: in->stride[0]: %d, in->stride[1]: %d, in->stride[2]: %d, in->stride[3]: %d\n", in->stride[0], in->stride[1], in->stride[2], in->stride[3]);
    printf("Strides: out->stride[0]: %d, out->stride[1]: %d, out->stride[2]: %d, out->stride[3]: %d\n", out->stride[0], out->stride[1], out->stride[2], out->stride[3]);

    printf("Extent: in->extent[0]: %d, in->extent[1]: %d, in->extent[2]: %d, in->extent[3]: %d\n", in->extent[0], in->extent[1], in->extent[2], in->extent[3]);
    printf("Extent: out->extent[0]: %d, out->extent[1]: %d, out->extent[2]: %d, in->extent[3]: %d\n", out->extent[0], out->extent[1], out->extent[2], out->extent[3]);
    */
    
    // Check the strides are what we want.
    assert(in->stride[0] == 1 && in->stride[1] == ch && in->stride[2] == w * ch && 
          out->stride[0] == 1 && out->stride[1] == ch && out->stride[2] == w * ch);

    //Get buffers
    float *in_buf = (float *)(in->host);
    float *out_buf = (float *)(out->host);
    float *param_buf = (float *)(params->host);
    
    Mat d(h, w, CV_32FC3, in_buf);
    Mat dNLM(h, w, CV_32FC3, out_buf);
    
    //Params
    float sigma_fixed = param_buf[0];
    float lambda_prior = param_buf[1];
    float sigma_scale = param_buf[2];
    int prior = (int)param_buf[3];
    
    //Fixed sigma if wanted
    float sigma_estim = sigma;
    if( sigma_fixed > 0.f)
        sigma_estim = sigma_fixed / 30 * sigma_scale;

    //Params
    cout << "Params are: [sigma =" << sigma_estim << ", lambda_prior=" << lambda_prior << ", prior="<< prior << "] !"<< endl;

    //Scale d
	double z_min, z_max;
	cv::minMaxIdx(d.reshape(1) , &z_min, &z_max);
	 
    //Scale and offset parameters d
    z_max = max(z_max, z_min + 0.01);
    double scale = 1.0/(z_max - z_min);
    double invscale = (z_max - z_min);

    //Offset
    d -= cv::Scalar(z_min,z_min,z_min);

    //###### Denoising params #######

    //Denoising params
    float sigma_luma = sigma_estim;
    float sigma_color = sigma_estim;
    if(prior == 1)
    	sigma_luma = 1.2 * sigma_estim; //NLM color stronger on luma
    
    //Template size
    int templateWindowSizeNLM = 5 * 2 + 1;
    int searchWindowSizeNLM = 1 * 2 + 1;

    //###### Do the denoising #######

	//Gpu
	GpuMat d_image(d);
	GpuMat d_image_uint;
	d_image.convertTo(d_image_uint, CV_8UC3, 255.0 * scale, 0.0);

	//GpuMat resNLM_uint;
	int64 start;
	if(prior == 0)
    {
        start = getTickCount();
		cuda::fastNlMeansDenoising(d_image_uint, d_image_uint, sigma_luma * 255.f, templateWindowSizeNLM, searchWindowSizeNLM);

        const double timeSec = (getTickCount() - start) / getTickFrequency();
        cout << "Fast NLM colored basic (CUDA): " << timeSec << " sec" << endl;
	}
	else if(prior == 1)
	{
	    start = getTickCount();
		cuda::fastNlMeansDenoisingColored(d_image_uint, d_image_uint, sigma_luma * 255.f, sigma_color * 255.f, templateWindowSizeNLM, searchWindowSizeNLM);

        const double timeSec = (getTickCount() - start) / getTickFrequency();
        cout << "Fast NLM colored CIELAB (CUDA): " << timeSec << " sec" << endl;
	}

	//Convert back and copy
	Mat resNLM_uint_host(d_image_uint); //Transfer back to cpu
	resNLM_uint_host.convertTo(dNLM, CV_32F, 1.0/255.0 * invscale);
      
    //Offset
    dNLM += cv::Scalar(z_min,z_min,z_min);

	return 0;
}
