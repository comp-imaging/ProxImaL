////////////////////////////////////////////////////////////////////////////////
//Convolution as part of image formation.
////////////////////////////////////////////////////////////////////////////////

#include <Halide.h>
using namespace Halide;
using namespace Halide::BoundaryConditions;

#include "core/image_formation.h"

//Full reduction
Func norm_L2(Func input, Expr width, Expr height, Expr ch, bool is1D=false)
{
	//is1D = false;
	//Square
	Func input_square("input_square");
	input_square(x, y, c) = input(x, y, c) * input(x, y, c) ;

	//Reduction
#if (1)
	Func resnorm("resnorm");
	if (!is1D) {
		int vec_width = 8;
		RDom r1(0, width), r2(0, height/vec_width), r3(0, vec_width, 0, ch);
		Func f1, f2;
		Var v;
		f1(v,y,c) = sum(input_square(r1,(y*vec_width)+v,c));
		f2(v,c) = sum(f1(v,r2,c));

		resnorm() = sqrt( sum(f2(r3.x, r3.y)) );

		//Schedule
		Var yo, yi;
		f1.split(y, yo, yi, 32).parallel(yo).compute_root();
		//f1.bound(v, 0, vec_width).vectorize(v, vec_width);
		f1.vectorize(v, vec_width);
		//f2.unroll(c,3).compute_root();
		//f2.compute_root();
		resnorm.compute_root();
	} else {
		int num_threads = 8;
		RDom r(0, width, 0, height/num_threads, 0, ch);
		Func partialNorm;
		Var x;
		partialNorm(x) = sum(input_square(r.x, (x*height/num_threads)+r.y, r.z));

		RDom rSeq(0, num_threads);
		resnorm() = sqrt( sum(partialNorm(rSeq.x)) );

		partialNorm.compute_root().parallel(x);
		resnorm.compute_root();
	}
#elif (0)
	RDom r1(0, width), r2(0, height), r3(0, ch);
	Func f1, f2;
	f1(y,c) = sum(input_square(r1,y,c));
	f2(c) = sum(f1(r2,c));

	Func resnorm("resnorm");
	resnorm() = sqrt( sum(f2(r3)) );

	//Schedule
	Var yo, yi;
	f1.split(y, yo, yi, 32).parallel(yo).compute_root();
	f2.unroll(c,3).compute_root();
	resnorm.compute_root();
#elif (0)
	RDom r(0, width, 0, height, 0, ch);
	Func resnorm;
	resnorm() = sqrt( sum(input_square(r.x, r.y, r.z)) );
#elif (0)
	int num_threads = 8;
	RDom r(0, width/num_threads, 0, height, 0, ch);
	Func partialNorm;
	Var x;
	partialNorm(x) = sum(input_square((x*width/num_threads)+r.x, r.y, r.z));

	RDom rSeq(0, num_threads);
	Func resnorm;
	resnorm() = sqrt( sum(partialNorm(rSeq.x)) );

	partialNorm.compute_root().parallel(x);

	resnorm.print_loop_nest();
#else
	int num_threads = 8;
	RDom r(0, width, 0, height/num_threads, 0, ch);
	Func partialNorm;
	Var x;
	partialNorm(x) = sum(input_square(r.x, (x*height/num_threads)+r.y, r.z));

	RDom rSeq(0, num_threads);
	Func resnorm;
	resnorm() = sqrt( sum(partialNorm(rSeq.x)) );

	partialNorm.compute_root().parallel(x);

	//resnorm.print_loop_nest();
#endif

	return resnorm;
}


class norm_L2_1D_gen : public Generator<norm_L2_1D_gen> {
public:

    ImageParam input{Float(32), 3, "input"};
    
    Func build() {
        Expr width = input.width();
        Expr height = input.height();
        Expr channels = input.channels();
        
        //Input
        Func input_func("in");
        input_func(x, y, c) = input(x, y, c);
        
        //Warping
        Func norm = norm_L2(input_func, width, height, channels, true);

        //Allow for arbitrary strides
        input.set_stride(0, Expr());

        return norm;
    }
};

class norm_L2_gen : public Generator<norm_L2_gen> {
public:

    ImageParam input{Float(32), 3, "input"};
    
    Func build() {
        Expr width = input.width();
        Expr height = input.height();
        Expr channels = input.channels();
        
        //Input
        Func input_func("in");
        input_func(x, y, c) = input(x, y, c);
        
        //Warping
        Func norm = norm_L2(input_func, width, height, channels, false); 

        //Allow for arbitrary strides
        input.set_stride(0, Expr());

        return norm;
    }
};

auto normL2_1DImg = RegisterGenerator<norm_L2_1D_gen>("normL2_1DImg");
auto normL2Img = RegisterGenerator<norm_L2_gen>("normL2Img");
