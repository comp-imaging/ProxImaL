////////////////////////////////////////////////////////////////////////////////
//This file contains all proximal operators that don't involve the image formation
//operators. Currently we have implemented L1 and the NLM prox operator from the
//FlexISP paper. Later, Poisson noise prox etc. will live here.
////////////////////////////////////////////////////////////////////////////////

#ifndef PROX_OPERATORS_H
#define PROX_OPERATORS_H

#include "vars.h"

////////////////////////////////////////////////////////////////////////////////
// Proximal L1
////////////////////////////////////////////////////////////////////////////////

//L1 anistotropic
Expr shrink_L1(Func v, Expr theta)
{
    return  max( 0.f, 1.f - theta/abs(v(x, y, c, k)) ) * v(x, y, c, k);
}

//L1 isotropic
Expr shrink_L1_iso(Func v, Expr theta)
{
    Expr norm = sqrt( v(x, y, c, 0) * v(x, y, c, 0)  + v(x, y, c, 1) * v(x, y, c, 1) );
    return  max( 0.f, 1.f - theta/norm ) * v(x, y, c, k) ;
}

//proxL1
Func proxIsoL1(Func input, Expr width, Expr height, Expr theta) {
    
	//Compute magnitude
    Func pInput("pInput");
    pInput(x, y, c, k) = shrink_L1_iso(input, theta);
        
    // Schedule
	pInput.parallel(y);
	pInput.compute_root();

    return pInput;
}

//proxL1
Func proxL1(Func input, Expr width, Expr height, Expr theta) {

    //Compute magnitude
    Func pInput("pInput");
    pInput(x, y, c, k) = shrink_L1(input, theta);
        
    // Schedule
	pInput.parallel(y);
    pInput.compute_root();

    return pInput;
}

////////////////////////////////////////////////////////////////////////////////
// Proximal operator poisson penalty with masking
////////////////////////////////////////////////////////////////////////////////

Expr poisson_penalty_masked(Func v, Func M, Func b, Expr theta)
{
    return  select( M(x, y, c) > 0.5f, 
                    0.5f * ( v(x, y, c) - theta + sqrt( (v(x, y, c) - theta)*(v(x, y, c) - theta) + 4.f * theta * b(x, y, c) ) ),
                    v(x, y, c) );
}


//proxL1 convex conjugate
Func proxPoisson(Func input, Expr width, Expr height, Func mask, Func b, Expr theta) {
    
    //Compute magnitude
    Func pInput("pInput");
    pInput(x, y, c) = poisson_penalty_masked(input, mask, b, theta);
        
    // Schedule
    pInput.parallel(y);
	pInput.compute_root();

    return pInput;
}

////////////////////////////////////////////////////////////////////////////////
// Proximal operator NLM
////////////////////////////////////////////////////////////////////////////////

//NLM extern with params
Func NLM(Func input, Expr sigma, Func params, Expr width, Expr height, Expr channels)
{
    Func NLM_ext("NLM_ext");
    std::vector < ExternFuncArgument > args = {
        input, params, sigma, width, height, channels
    };
    NLM_ext.define_extern("NLM_extern", args, Float(32), 3);
    
    return NLM_ext;
}

Func proxNLM(Func v, Expr theta, Func params_func, Expr width, Expr height, Expr channels)
{
    Func v_flip("v_flip");
    v_flip(c, x, y) = v(x, y, c); //ASSUMES NORMALIZATAION IN NLM
    
    //Schedule
    v_flip.compute_root();
    params_func.compute_root();

    Func NLM_v("NLM_v");
    NLM_v = NLM(v_flip, sqrt(theta), params_func, width, height, channels);
    NLM_v.compute_root();

    //Reshape
    Func NLM_res("NLM_res");
    NLM_res(x, y, c) = NLM_v(c, x, y);

    return NLM_res;
}

////////////////////////////////////////////////////////////////////////////////
// FANN patch-wise denoising
////////////////////////////////////////////////////////////////////////////////

//FANN extern with params
Func FANN(Func input, Expr sigma, Func params, Expr verbose, Expr width, Expr height, Expr channels, Expr numparams)
{
    Func FANN_ext("FANN_ext");
    std::vector < ExternFuncArgument > args = {
        input, params, sigma, verbose, width, height, channels, numparams
    };
    FANN_ext.define_extern("FANN_extern", args, Float(32), 3);
    
    return FANN_ext;
}

Func proxFANN(Func v, Expr sigma, Func params_func, Expr verbose, Expr width, Expr height, Expr channels, Expr numparams)
{
    //Schedule
    v.compute_root();
    params_func.compute_root();

    Func FANN_v("FANN_v");
    FANN_v = FANN(v, sigma, params_func, verbose, width, height, channels, numparams);
    FANN_v.compute_root();

    //Reshape (boundary inference)
    Func FANN_res("FANN_res");
    FANN_res(x, y, c) = FANN_v(x, y, c);

    return FANN_res;
}


#endif //PROX_OPERATORS
