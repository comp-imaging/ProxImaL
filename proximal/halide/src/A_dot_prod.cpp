////////////////////////////////////////////////////////////////////////////////
//Convolution as part of image formation.
////////////////////////////////////////////////////////////////////////////////

#include <Halide.h>
using namespace Halide;
using namespace Halide::BoundaryConditions;

#include "core/image_formation.h"

//Full reduction
Func dot_prod(Func A, Func B, Expr width, Expr height, Expr ch, bool is1D=false)
{
	Func input_square("input_square");
	input_square(x, y, c) = A(x, y, c) * B(x, y, c) ;

	//is1D = false;
#if (1)
	Func result("resdot");
	if (!is1D) {
		int vec_width = 8;
		RDom r1(0, width), r2(0, height/vec_width), r3(0, vec_width, 0, ch);
		Func f1, f2;
		Var v;
		f1(v,y,c) = sum(input_square(r1,(y*vec_width)+v,c));
		f2(v,c) = sum(f1(v,r2,c));

		result() = sum(f2(r3.x, r3.y));

		//Schedule
		Var yo, yi;
		f1.split(y, yo, yi, 32).parallel(yo).compute_root();
		//f1.bound(v, 0, vec_width).vectorize(v, vec_width);
		f1.vectorize(v, vec_width);
		//f2.unroll(c,3).compute_root();
		//f2.compute_root();
		result.compute_root();
	} else {
		int num_threads = 8;
		RDom r(0, width, 0, height/num_threads, 0, ch);
		Func partialSum;
		Var x;
		partialSum(x) = sum(input_square(r.x, (x*height/num_threads)+r.y, r.z));

		RDom rSeq(0, num_threads);
		result() =  sum(partialSum(rSeq.x));

		partialSum.compute_root().parallel(x);
		result.compute_root();
	}
#endif
#if (0)

	RDom r(0, width, 0, height, 0, ch);
	Func result("resdot");
	result() = ( sum(input_square(r.x, r.y, r.z)) );
#endif
#if (0)
	//Vectorization
	const int vec_size = 8;
	std::cout << "Vectorizing dot product for size: " <<  vec_size << std::endl;

	//Vectorization
	Var v("v"), l("l");

	//Dims
	Expr size = width * height * ch;

	//Input
	Func A_lin("A_lin");
	A_lin(l) = A(l % width, (l / width) % height,  l / (width * height) );

	Func B_lin("B_lin");
	B_lin(l) = B(l % width, (l / width) % height,  l / (width * height) );

	//Dot product
	Expr size_vecs = size / vec_size;
	Expr size_tail = size - size_vecs * vec_size;

	//Func result;
	Func dot;
	RDom k(0, size_vecs);
	dot(v) = sum( A_lin(k*vec_size + v) * B_lin(k*vec_size + v) );

	RDom lanes(0, vec_size);
	RDom tail(size_vecs * vec_size, size_tail);

	Func result("result");
	result() = sum(dot(lanes)) + sum(A_lin(tail) * B_lin(tail));

	//Schedule
	A.compute_root();
	B.compute_root();
	dot.compute_root().vectorize(v);
	result.compute_root();
#endif

	return result;
}


class dot_prod_1D_gen : public Generator<dot_prod_1D_gen> {
public:

    ImageParam A{Float(32), 3, "A"};
    ImageParam B{Float(32), 3, "B"};
    
	Func build() {
        Expr width = A.width();
        Expr height = A.height();
        Expr channels = A.channels();
        
        //Input
        Func A_func("A_func");
        A_func(x, y, c) = A(x, y, c);
        
		Func B_func("B_func");
        B_func(x, y, c) = B(x, y, c);
       
		//Dot product
		Func dotAB = dot_prod( A_func, B_func, width, height, channels, true );

        //Allow for arbitrary strides
        A.set_stride(0, Expr());
        B.set_stride(0, Expr());

        return dotAB;
    }
};

class dot_prod_gen : public Generator<dot_prod_gen> {
public:

    ImageParam A{Float(32), 3, "A"};
    ImageParam B{Float(32), 3, "B"};
    
	Func build() {
        Expr width = A.width();
        Expr height = A.height();
        Expr channels = A.channels();
        
        //Input
        Func A_func("A_func");
        A_func(x, y, c) = A(x, y, c);
        
		Func B_func("B_func");
        B_func(x, y, c) = B(x, y, c);
       
		//Dot product
		Func dotAB = dot_prod( A_func, B_func, width, height, channels, false);

        //Allow for arbitrary strides
        A.set_stride(0, Expr());
        B.set_stride(0, Expr());

        return dotAB;
    }
};

auto dot_1DImg = RegisterGenerator<dot_prod_1D_gen>("dot_1DImg");
auto dotImg = RegisterGenerator<dot_prod_gen>("dotImg");
