////////////////////////////////////////////////////////////////////////////////
//This file contains all the linear operators for the image formation.
//The image formation A is a concatenation of matrices. In this file we have implemented
// A = M * K (convolution via Toeplitz matrix K forllowed by diagonal weighting matrix M)
// A = M * W (warping matrix followed by diagonal weighting matrix M)
//
// We have explicitly implemented A^T * A here, which is used in the CG. In general we need 
// A, A^T and A^T * A implemented. A^T * A is most efficient when fused, but does not have to for now. 
////////////////////////////////////////////////////////////////////////////////

#ifndef IMAGE_FORMATION_H
#define IMAGE_FORMATION_H

#include "vars.h"

////////////////////////////////////////////////////////////////////////////////
// Convolution
////////////////////////////////////////////////////////////////////////////////

//Convolution
Func A_conv(Func input, Expr width, Expr height, Func K, Expr filter_width, Expr filter_height) {

    //Clamped
    Func img_bounded("img_bounded");
    img_bounded = BoundaryConditions::repeat_image(input, 0, width, 0, height);

    //Define the convolution
    Func img_conv("img_conv");
    RDom rf(0, filter_width, 0, filter_height);
    img_conv(x, y, c) = sum(img_bounded(x - rf.x + filter_width / 2, y - rf.y + filter_height / 2, c) * K(rf.x, rf.y, c));	

    std::cout << "Finished A_conv setup." << std::endl;

    // Schedule
    K.compute_root();
    img_conv.reorder(c, x, y);

    // Parallel
    img_conv.vectorize(x, 32);
    img_conv.parallel(y, 8);

    // Apply the boundary condition to the input as required
    img_bounded.compute_at(img_conv, y);
    
    return img_conv;
}

//At via convolution
Func At_conv(Func input, Expr width, Expr height, Func K, Expr filter_width, Expr filter_height) {

    //Clamped
    Func img_bounded("img_bounded");
    img_bounded = BoundaryConditions::repeat_image(input, 0, width, 0, height);

    //Define the convolution
    Func img_conv("img_conv");
    RDom rf(0, filter_width, 0, filter_height);
    img_conv(x, y, c) = sum(img_bounded(x - rf.x + filter_width / 2, y - rf.y + filter_height / 2, c) * K(filter_width - 1 - rf.x, filter_height - 1 - rf.y, c)); 

    std::cout << "Finished At_conv setup." << std::endl;

    // Schedule
    K.compute_root();
    img_conv.reorder(c, x, y);

    // Parallel
    img_conv.vectorize(x, 32);
    img_conv.parallel(y, 8);

    // Apply the boundary condition to the input as required
    img_bounded.compute_at(img_conv, y);
    
    return img_conv;
}

////////////////////////////////////////////////////////////////////////////////
// Mask
////////////////////////////////////////////////////////////////////////////////

//Mask diagonal weighting matrix
Func A_M(Func input, Expr width, Expr height, Func mask) {

	int vec_width = 8;
    //Define the mask
    Func input_mask("input_mask");
    input_mask(x, y, c) = mask(x, y, c) * input(x, y, c);

    std::cout << "Finished A_mask setup." << std::endl;

	// Schedule
    input_mask.reorder(c, y, x);
    input_mask.vectorize(y, vec_width);
	//input_mask.unroll(c,3);
    input_mask.parallel(x);

	//input_mask.print_loop_nest();

    return input_mask;
}

//Diagonal matrix (so At_M = A_M)
Func At_M(Func input, Expr width, Expr height, Func mask) {
    return A_M(input, width, height, mask);
}

//Mask application
Func AtA_M(Func input, Expr width, Expr height, Func mask) {

    int vec_width = 8;

	//Define the mask
    Func input_mask("input_mask");
    input_mask(x, y, c) = mask(x, y, c) * mask(x, y, c) * input(x, y, c);

    std::cout << "Finished AtA_mask setup." << std::endl;

    // Schedule
    input_mask.reorder(c, y, x);
    input_mask.vectorize(y, vec_width);
	input_mask.parallel(x);

	//input_mask.unroll(c,3);

	input_mask.print_loop_nest();


    
    return input_mask;
}

////////////////////////////////////////////////////////////////////////////////
// Warp
////////////////////////////////////////////////////////////////////////////////

// Interpolation kernel (linear)
Expr kernel_linear(Expr x) {
    Expr xx = abs(x);
    return select(xx < 1.0f, 1.0f - xx, 0.0f);
}

//At MtM A via warping
Func A_warpHomography(Func input, Expr width, Expr height, Func H, Expr nhom) {

    //Local vars
    Var xo, xi;

     //Clamped
    Func clamped("clampedInput");
    clamped = constant_exterior(input, 0.f, 0, width, 0, height);
    
    //coords from homography
    Expr n = x * H(2,0,g) + y * H(2,1,g) + H(2,2,g);
    Expr sourcex = (x * H(0,0,g) + y * H(0,1,g) + H(0,2,g))/n ;
    Expr sourcey = (x * H(1,0,g) + y * H(1,1,g) + H(1,2,g))/n ;
    
    // Initialize interpolation kernels. 
    Func kernel("kernel");
    Expr beginx = cast<int>(sourcex);
    Expr beginy = cast<int>(sourcey);
    RDom dom(0, 2, 0, 2, "dom");
    kernel(x,y,g,k,l) = kernel_linear(k + beginx - sourcex) * kernel_linear(l + beginy - sourcey);

    // Perform resampling
    Func resampled("resampled");
    resampled(x, y, c, g) = sum(kernel(x,y,g, dom.x, dom.y) * cast<float>(clamped(dom.x + beginx, dom.y + beginy, c)));
    
    //Schedule backprojection
        // Scheduling
    bool parallelize = 1;
    bool vectorize = 1;
    kernel.compute_at(resampled, x);
    
    if (vectorize) {
		int vec_size = 8;
        resampled.vectorize(y, vec_size);
    }

    if (parallelize) {
        Var xo, xi;
        resampled.split(x, xo, x, 32).parallel(xo);
    }
    resampled.compute_root();
    
    return resampled;
}

Func At_warpHomography(Func input, Expr width, Expr height, Func Hinv, Expr nhom) {

    //Local vars
    Var xo, xi;

    //Get constand boundary
    Func clamped("clampedInput");
    clamped = constant_exterior(input, 0.f, 0, width, 0, height);
    
    //coords from second homography
    Expr n = x * Hinv(2,0,g) + y * Hinv(2,1,g) + Hinv(2,2,g);
    Expr sourcex = (x * Hinv(0,0,g) + y * Hinv(0,1,g) + Hinv(0,2,g))/n;
    Expr sourcey = (x * Hinv(1,0,g) + y * Hinv(1,1,g) + Hinv(1,2,g))/n;
    
    // Initialize interpolation kernels. 
    Func kernel("kernel");
    Expr beginx = cast<int>(sourcex);
    Expr beginy = cast<int>(sourcey);
    RDom dom(0, 2, 0, 2, "dom");
    kernel(x,y,g,k,l) = kernel_linear(k + beginx - sourcex) * kernel_linear(l + beginy - sourcey);

    // Perform resampling
    Func resampledAt("resampledAt");
    resampledAt(x, y, c, g) = sum(kernel(x,y,g, dom.x, dom.y) * clamped(dom.x + beginx, dom.y + beginy, c, g));
    
    //Func final
    RDom domH(0, nhom, "domH");
    Func resampledAtSum("resampledAtSum");
    resampledAtSum(x, y, c) = sum(resampledAt(x, y, c, domH));

    std::cout << "Finished At_warp setup." << std::endl;

    // Scheduling
    bool parallelize = 1;
    bool vectorize = 1;
    kernel.compute_at(resampledAt, x);
    
    if (vectorize) {
        resampledAt.vectorize(y, 8);
    }

    if (parallelize) {
        Var xo, xi;
        resampledAt.split(x, xo, x, 32).parallel(xo);
    }
    resampledAt.compute_root();

    //Final sum
    if (vectorize) {
        resampledAtSum.vectorize(y, 8);
    }

    if (parallelize) {
        Var xo, xi;
        resampledAtSum.split(x, xo, x, 32).parallel(xo);
    }
    resampledAtSum.compute_root();

    return resampledAtSum;
}

#endif //IMAGE_FORMATION_H
