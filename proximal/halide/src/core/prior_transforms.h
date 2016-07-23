////////////////////////////////////////////////////////////////////////////////
// This file includes the linear operators in objective. In our case here it is
// K = [ Identity;
//       dx;
//       dy];
//
// This allows for TV priors via sparse shrinkage. Any stack of matrices possible here.
// The file contains the matrix K and its adjoint.
////////////////////////////////////////////////////////////////////////////////

#ifndef PRIOR_TRANSFORMS_H
#define PRIOR_TRANSFORMS_H

#include "vars.h"

//K gradient mat func
// Here K is only the stacked gradients
//
// K = [ dx;
//       dy];
Func K_grad_mat(Func input, Expr width, Expr height) {
    
	int vec_width = 8;

    Func Kx("Kx");

    // Compute gradient
    Func inBounded("inBounded");
    inBounded = mirror_image(input, 0, width, 0, height);
    
	Func dx("dx");
	Func dy("dy");
	dx(x, y, c) =  inBounded(x + 1, y, c) - inBounded(x, y, c);
    dy(x, y, c) =  inBounded(x, y + 1, c) - inBounded(x, y, c);
	
	Kx(x, y, c, k) = select(k == 0, dy(x, y, c), dx(x, y, c));

    // Schedule
	Kx.reorder(k, c, y, x);
	Kx.vectorize(y, vec_width);
	Kx.parallel(x);
	Kx.unroll(k,2);
	Kx.compute_root();

	//Kx.print_loop_nest();

    return Kx;
}

//KT just for gradient
Func KT_grad_mat(Func input, Expr width, Expr height) {
   
	int vec_width = 8;
    
	Func KTp("KTp");
    Func KTx("KTx");
    Func KTy("KTy");
    Func KT("KT");
    
    // Compute gradient for current iteration
    Func inBounded("inBounded");
    inBounded = mirror_image(input, 0, width, 0, height);

	KTy(x, y, c) = select(
			y == 0, inBounded(x, 0, c, 0),
			y == height-1, -inBounded(x, height - 2, c, 0),
			inBounded(x, y, c, 0) - inBounded(x, y - 1, c, 0)
			);

	KTx(x, y, c) =   select(
			x == 0,  inBounded(0, y, c, 1),
			x == width-1, -inBounded(width - 2, y, c, 1),
			inBounded(x, y, c, 1) - inBounded(x - 1, y, c, 1)
			);
	
    //Final result is sum of all matrix-vector products
    KT(x, y, c) = -KTx(x, y, c) - KTy(x, y, c);

    std::cout << "Finished KT setup." << std::endl;
    
	// Schedule
	KT.reorder(c, y, x);
	KT.parallel(x);
	KT.vectorize(y, vec_width);
	KT.compute_root();
	
	//KT.print_loop_nest();
	
    return KT;
}

/*
// Kt * K just for gradients
Func KTK_grad_mat(Func input, Expr width, Expr height) {

    Func Kx("Kx");
    Func KTx("KTx");
    Func KTy("KTy");
    Func KT("KT");
    
    // Pure definition: do nothing.
    Kx(x, y, c, k) = undef<float>();
    
    // Compute gradient K
    Func inBounded("inBounded");
    inBounded = mirror_image(input, 0, width, 0, height);
    Kx(x, y, c, 1) =  inBounded(x + 1, y, c) - inBounded(x, y, c);
    Kx(x, y, c, 2) =  inBounded(x, y + 1, c) - inBounded(x, y, c);

     // Compute gradient transpose Kt
    Func KxBounded("KxBounded");
    KxBounded = mirror_image(Kx, 0, width, 0, height);
    
    KTx(x, y, c) =   KxBounded(x, y, c, 1) - KxBounded(x - 1, y, c, 1) ;
    KTx(0, y, c) =   KxBounded(0, y, c, 1) ;
    KTx(width - 1, y, c) = -KxBounded(width - 2, y, c, 1) ;
    
    KTy(x, y, c) =  KxBounded(x, y, c, 2) - KxBounded(x, y - 1, c, 2);
    KTy(x, 0, c) =  KxBounded(x, 0, c, 2);
    KTy(x, height - 1, c) =  -KxBounded(x, height - 2, c, 2);
    
    //Final result is sum of all matrix-vector products
    KT(x, y, c) = - KTx(x, y, c) - KTy(x, y, c);

    // Schedule
    Kx.split(x, xo, xi, 16);
    Kx.unroll(xi,16);
    Kx.compute_root();
    
    // Schedule
    KTx.split(x, xo, xi, 16);
    KTx.unroll(xi,16);
    KTx.compute_root();
    
    // Schedule
    KTy.split(x, xo, xi, 16);
    KTy.unroll(xi,16);
    KTy.compute_root();

    return KT;
}
*/

/*
//K mat func
Func K_mat(Func input, Expr width, Expr height, Expr lambda_prior, Expr lambda_grad) {
    
    // Pure definition: do nothing.
    Func Kx("Kx");
    Kx(x, y, c, k) = undef<float>();
    
    // Compute gradient
    Func inBounded("inBounded");
    inBounded = mirror_image(input, 0, width, 0, height);
    
    Kx(x, y, c, 0) =  lambda_prior * inBounded(x, y, c);
    Kx(x, y, c, 1) =  lambda_grad * ( inBounded(x + 1, y, c) - inBounded(x, y, c) );
    Kx(x, y, c, 2) =  lambda_grad * ( inBounded(x, y + 1, c) - inBounded(x, y, c) );

    // Schedule
    Kx.split(x, xo, xi, 16);
    Kx.unroll(xi,16);
    Kx.compute_root();

    return Kx;
}


//KT mat func
Func KT_mat(Func input, Expr width, Expr height, Expr lambda_prior, Expr lambda_grad) {
    
    Func KTp("KTp");
    Func KTx("KTx");
    Func KTy("KTy");
    Func KT("KT");
    
    // Compute gradient for current iteration
    Func inBounded("inBounded");
    inBounded = mirror_image(input, 0, width, 0, height);
    
    KTp(x, y, c) = inBounded(x, y, c, 0);
    
    KTx(x, y, c) =   inBounded(x, y, c, 1) - inBounded(x - 1, y, c, 1) ;
    KTx(0, y, c) =   inBounded(0, y, c, 1) ;
    KTx(width - 1, y, c) = -inBounded(width - 2, y, c, 1) ;
    
    KTy(x, y, c) =  inBounded(x, y, c, 2) - inBounded(x, y - 1, c, 2);
    KTy(x, 0, c) =  inBounded(x, 0, c, 2);
    KTy(x, height - 1, c) =  -inBounded(x, height - 2, c, 2);
    
    //Final result is sum of all matrix-vector products
    KT(x, y, c) = lambda_prior * KTp(x, y, c) + (-lambda_grad * KTx(x, y, c) ) + (-lambda_grad * KTy(x, y, c) );
    
    // Schedule
    KTx.split(x, xo, xi, 16);
    KTx.unroll(xi,16);
    KTx.compute_root();
    
    // Schedule
    KTy.split(x, xo, xi, 16);
    KTy.unroll(xi,16);
    KTy.compute_root();

    return KT;
}
*/


#endif //PRIOR_TRANSFORMS_H
