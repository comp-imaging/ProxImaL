////////////////////////////////////////////////////////////////////////////////
//FANN proximal operator from "core/prox_operators.h"
////////////////////////////////////////////////////////////////////////////////

#include <Halide.h>

using namespace Halide;
using namespace Halide::BoundaryConditions;

#include "core/prox_operators.h"

class proxFANN_extern_gen : public Generator<proxFANN_extern_gen> {
public:

    ImageParam input{Float(32), 3, "input"};
    Param<float> sigma{"sigma"};
    ImageParam params{Float(32), 2, "params"};
    Param<int> verbose{"verbose"};

    Func build() {

        Expr width = input.width();
        Expr height = input.height();
        Expr channels = input.channels();
        Expr numparams = params.width();
     
        //Get input and reshuffle
        Func input_func("input_func");
        input_func(x, y, c) = input(x, y, c);
        
        //Params
        Func params_func("param_func");
        params_func(k) = params(0,k);

        //Schedule
        Func FANN_input("FANN_input");
        FANN_input = proxFANN(input_func, sigma, params_func, verbose, width, height, channels, numparams);
        FANN_input.compute_root();

        //Allow for arbitrary strides
        input.set_stride(0, Expr());
        params.set_stride(0, Expr());
        FANN_input.output_buffer().set_stride(0, Expr()); 
        
        return FANN_input;
    }
};

auto proxFANNextern = RegisterGenerator<proxFANN_extern_gen>("proxFANN_extern_gen");