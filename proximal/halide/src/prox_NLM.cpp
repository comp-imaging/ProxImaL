////////////////////////////////////////////////////////////////////////////////
//NLM proximal operator from "core/prox_operators.h"
////////////////////////////////////////////////////////////////////////////////

#include <Halide.h>

using namespace Halide;
using namespace Halide::BoundaryConditions;

#include "core/prox_operators.h"

class proxNLM_extern_gen : public Generator<proxNLM_extern_gen> {
public:

    ImageParam input{Float(32), 3, "input"};
    Param<float> theta{"theta"};
    ImageParam params{Float(32), 2, "params"};

    Func build() {

        Expr width = input.width();
        Expr height = input.height();
        Expr channels = input.channels();
     
        //Get input and reshuffle
        Func input_func("input_func");
        input_func(x, y, c) = input(x, y, c);
        
        //Params
        Func params_func("param_func");
        params_func(k) = params(0,k);

        //Schedule
        Func NLM_input("NLM_input");
        NLM_input = proxNLM(input_func, theta, params_func, width, height, channels);
        NLM_input.compute_root();

        //Allow for arbitrary strides
        input.set_stride(0, Expr());
        params.set_stride(0, Expr());
        NLM_input.output_buffer().set_stride(0, Expr()); 
        
        return NLM_input;
    }
};

auto proxNLMextern = RegisterGenerator<proxNLM_extern_gen>("proxNLM_extern_gen");