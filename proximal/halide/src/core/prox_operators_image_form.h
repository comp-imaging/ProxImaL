////////////////////////////////////////////////////////////////////////////////
//This file contains all proximal operators involving the image formation operators.
//Currently all image formation proximal operators are computed via CG. The two CG 
//implementations in this file are identical except for the function call of A^T*A,
//where A is here the concatenation of all image formation matrices.
////////////////////////////////////////////////////////////////////////////////

#ifndef PROX_OPERATORS_IMAGE_FORM_H
#define PROX_OPERATORS_IMAGE_FORM_H

//Vars
#include "vars.h"

//Image formation
#include "image_formation.h"

//Image formation
#include "prior_transforms.h"

//CG utils
#include "CG_utils.h"

////////////////////////////////////////////////////////////////////////////////
//proximal operator (based on conjugate gradient) for convolution + mask
////////////////////////////////////////////////////////////////////////////////

//CG with iteration labels
std::vector<Func>  CG_conv(Func xin_func, Func Atb, Func K_func, Func MtM_func, 
        Expr width, Expr height, Expr ch, Expr width_kernel, Expr height_kernel, 
        Expr beta_cg, 
        int cgiters, int outter_iter)
{
        //Outer iteration number
        std::string oitstr = std::to_string(outter_iter);

        //Compute b
        // b = beta_cg * AtB + f
        Func b_func("b_cg" + oitstr);
        b_func(x, y, c) = beta_cg * Atb(x, y, c) + xin_func(x, y, c);

        // Compute residual
        // r = b - Mfun(x)
        Func r("r_cg" + oitstr);
        Func AtA_x("AtA_x_cg" + oitstr);
        AtA_x = AtA_conv(xin_func, width, height, K_func, width_kernel, height_kernel, MtM_func);

        Func Mfun_x("Mfun_x_" + oitstr);
        Mfun_x(x, y, c) = beta_cg * AtA_x(x, y, c) + xin_func(x, y, c);

        r(x, y, c) = b_func(x, y, c) - Mfun_x(x, y, c);

        
        //Schedule r
        r.vectorize(x, 4);
        Var yo, yi;
        r.split(y, yo, y, 32).parallel(yo);
        r.compute_root();
        
        ///////////////////////////////////////////////////////////////////
        // Do iterations
        ///////////////////////////////////////////////////////////////////
        
        //Iterates (iteration plus potentially initial value)
        Func rho[cgiters + 1];
        Func p[cgiters + 1];
        Func r_func[cgiters + 1];
        Func x_func[cgiters + 1];

        //Intial iterate and residual
        x_func[0] = xin_func;
        r_func[0] = r;

        for( int it = 1; it <= cgiters; it++ )
        {
            //Output
            std::string itstr = std::to_string(it);
            std::cout << "Scheduling iteration [" << it << "/" << cgiters << "]" << std::endl;

            //Compute rho
            // rho = (r(:)'*r(:));
            rho[it] = Func( "rho_cg_" + itstr + "_" + oitstr );
            rho[it]() = dot_prod(r_func[it - 1], r_func[it - 1], width, height, ch);

            //Compute p
            p[it] = Func("p_cg_" + itstr + "_" + oitstr  );
            if( it == 1 )
            {
                // p = r
                p[it](x, y, c) = r_func[it - 1](x, y, c);
            }
            else
            {
                // p = r + (rho / rho_1)*p;
                p[it](x, y, c) = r_func[it - 1](x, y, c) + ( rho[it]() / rho[it - 1]() ) * p[it - 1](x, y, c);
            }
            
            //Compute q = Ap
            //q = Mfun(p);
            Func AtA_p("AtA_p_cg_" + itstr + "_" + oitstr );
            AtA_p = AtA_conv(p[it], width, height, K_func, width_kernel, height_kernel, MtM_func);

            Func q("q_cg_" + itstr + "_" + oitstr  );
            q(x, y, c) = beta_cg * AtA_p(x, y, c) + p[it](x, y, c);


            
            //Compute alpha
            //alpha = rho / (p(:)'*q(:) );
            Func pq("pq_cg_" + itstr + "_" + oitstr  );
            pq() = dot_prod(p[it], q, width, height, ch);
            
            Func alpha("alpha_cg_" + itstr  + "_" + oitstr );
            alpha() = rho[it]() / pq();
            
            //Compute x
            //x = x + alpha * p; % update approximation vector
            x_func[it] = Func("xout_cg_" + itstr + "_" + oitstr );
            x_func[it](x,y,c) = x_func[it - 1](x,y,c) + alpha() * p[it](x,y,c);
            
            //Compute r
            //r = r - alpha*q;  ; % compute residual
            r_func[it] = Func("rout_cg_" + itstr + "_" + oitstr );
            r_func[it](x,y,c) = r_func[it - 1](x,y,c) - alpha() * q(x,y,c);
            
            //Schedule r
            x_func[it].compute_root();
            r_func[it].compute_root();
        }

        //Compute the norm of r for final iterate
        Func norm_r("norm_r");
        norm_r() = norm_L2(r_func[cgiters], width, height, ch);

        std::vector<Func> outputs;
        outputs.push_back(x_func[cgiters]); 
        outputs.push_back(norm_r);     
        
        //Compute the norm of r for final iterate
        //return x_func[cgiters];
        return outputs;
}

////////////////////////////////////////////////////////////////////////////////
//proximal operator (based on conjugate gradient) for warp + mask
////////////////////////////////////////////////////////////////////////////////

//CG with iteration labels
std::vector<Func>  CG_warp(Func xin_func, Func Atb, Func H_func, Func Hinv_func, Func MtM_func, 
        Expr width, Expr height, Expr ch, Expr nhom, 
        Expr beta_cg, 
        int cgiters, int outter_iter)
{
        //Outer iteration number
        std::string oitstr = std::to_string(outter_iter);

        //Compute b
        // b = beta_cg * AtB + f
        Func b_func("b_cg" + oitstr);
        b_func(x, y, c) = beta_cg * Atb(x, y, c) + xin_func(x, y, c);

        // Compute residual
        // r = b - Mfun(x)
        Func r("r_cg" + oitstr);
        Func AtA_x("AtA_x_cg" + oitstr);
        AtA_x = AtA_warpHomography(xin_func, width, height, H_func, Hinv_func, nhom, MtM_func);
        Func Mfun_x("Mfun_x_" + oitstr);
        Mfun_x(x, y, c) = beta_cg * AtA_x(x, y, c) + xin_func(x, y, c);

        r(x, y, c) = b_func(x, y, c) - Mfun_x(x, y, c);

        
        //Schedule r
        r.vectorize(x, 4);
        Var yo, yi;
        r.split(y, yo, y, 32).parallel(yo);
        r.compute_root();
        
        ///////////////////////////////////////////////////////////////////
        // Do iterations
        ///////////////////////////////////////////////////////////////////
        
        //Iterates (iteration plus potentially initial value)
        Func rho[cgiters + 1];
        Func p[cgiters + 1];
        Func r_func[cgiters + 1];
        Func x_func[cgiters + 1];

        //Intial iterate and residual
        x_func[0] = xin_func;
        r_func[0] = r;

        for( int it = 1; it <= cgiters; it++ )
        {
            //Output
            std::string itstr = std::to_string(it);
            std::cout << "Scheduling iteration [" << it << "/" << cgiters << "]" << std::endl;

            //Compute rho
            // rho = (r(:)'*r(:));
            rho[it] = Func( "rho_cg_" + itstr + "_" + oitstr );
            rho[it]() = dot_prod(r_func[it - 1], r_func[it - 1], width, height, ch);

            //Compute p
            p[it] = Func("p_cg_" + itstr + "_" + oitstr  );
            if( it == 1 )
            {
                // p = r
                p[it](x, y, c) = r_func[it - 1](x, y, c);
            }
            else
            {
                // p = r + (rho / rho_1)*p;
                p[it](x, y, c) = r_func[it - 1](x, y, c) + ( rho[it]() / rho[it - 1]() ) * p[it - 1](x, y, c);
            }
            
            //Compute q = Ap
            //q = Mfun(p);
            Func AtA_p("AtA_p_cg_" + itstr + "_" + oitstr );
            AtA_p = AtA_warpHomography(p[it], width, height, H_func, Hinv_func, nhom, MtM_func);

            Func q("q_cg_" + itstr + "_" + oitstr  );
            q(x, y, c) = beta_cg * AtA_p(x, y, c) + p[it](x, y, c);


            
            //Compute alpha
            //alpha = rho / (p(:)'*q(:) );
            Func pq("pq_cg_" + itstr + "_" + oitstr  );
            pq() = dot_prod(p[it], q, width, height, ch);
            
            Func alpha("alpha_cg_" + itstr  + "_" + oitstr );
            alpha() = rho[it]() / pq();
            
            //Compute x
            //x = x + alpha * p; % update approximation vector
            x_func[it] = Func("xout_cg_" + itstr + "_" + oitstr );
            x_func[it](x,y,c) = x_func[it - 1](x,y,c) + alpha() * p[it](x,y,c);
            
            //Compute r
            //r = r - alpha*q;  ; % compute residual
            r_func[it] = Func("rout_cg_" + itstr + "_" + oitstr );
            r_func[it](x,y,c) = r_func[it - 1](x,y,c) - alpha() * q(x,y,c);
            
            //Schedule r
            x_func[it].compute_root();
            r_func[it].compute_root();
        }

        //Compute the norm of r for final iterate
        Func norm_r("norm_r");
        norm_r() = norm_L2(r_func[cgiters], width, height, ch);

        std::vector<Func> outputs;
        outputs.push_back(x_func[cgiters]); 
        outputs.push_back(norm_r);     
        
        //Compute the norm of r for final iterate
        //return x_func[cgiters];
        return outputs;
}


////////////////////////////////////////////////////////////////////////////////
//proximal operator (based on conjugate gradient) for convolution + mask for quadratic ADMM
////////////////////////////////////////////////////////////////////////////////


//CG with iteration labels
std::vector<Func>  CG_conv_quadratic(Func xin_func, Func xi_K, Func xi_grad, Func K_func, Func M_func, Func MtM_func, 
        Expr width, Expr height, Expr ch, Expr width_kernel, Expr height_kernel, 
        Expr beta_cg, 
        int cgiters, int outter_iter)
{
        //Outer iteration number
        std::string oitstr = std::to_string(outter_iter);

        //Compute AtB
        Func AtB_xi_K("AtB_xi_K" + oitstr);
        AtB_xi_K = At_conv(xi_K, width, height, K_func, width_kernel, height_kernel, M_func);

        Func AtB_xi_grad("AtB_xi_grad" + oitstr);
        AtB_xi_grad = KT_grad_mat(xi_grad, width, height);

        //Compute b
        // b = At(xi_K) + beta_cg * Kt(x_grad)
        Func b_func("b_cg" + oitstr);
        b_func(x, y, c) = AtB_xi_K(x, y, c) + beta_cg * AtB_xi_grad(x, y, c);


        // Compute residual
        // r = b - Mfun(x)
        Func r("r_cg" + oitstr);
        Func AtA_x("AtA_x_cg" + oitstr);
        AtA_x = AtA_conv(xin_func, width, height, K_func, width_kernel, height_kernel, MtM_func);

        Func KtK_x("KtK_x_cg" + oitstr);
        KtK_x = KTK_grad_mat(xin_func, width, height);

        Func Mfun_x("Mfun_x_" + oitstr);
        Mfun_x(x, y, c) = AtA_x(x, y, c) + beta_cg * KtK_x(x, y, c);

        r(x, y, c) = b_func(x, y, c) - Mfun_x(x, y, c);
        
        //Schedule r
        r.vectorize(x, 4);
        Var yo, yi;
        r.split(y, yo, y, 32).parallel(yo);
        r.compute_root();
        
        ///////////////////////////////////////////////////////////////////
        // Do iterations
        ///////////////////////////////////////////////////////////////////
        
        //Iterates (iteration plus potentially initial value)
        Func rho[cgiters + 1];
        Func p[cgiters + 1];
        Func r_func[cgiters + 1];
        Func x_func[cgiters + 1];

        //Intial iterate and residual
        x_func[0] = xin_func;
        r_func[0] = r;

        for( int it = 1; it <= cgiters; it++ )
        {
            //Output
            std::string itstr = std::to_string(it);
            std::cout << "Scheduling iteration [" << it << "/" << cgiters << "]" << std::endl;

            //Compute rho
            // rho = (r(:)'*r(:));
            rho[it] = Func( "rho_cg_" + itstr + "_" + oitstr );
            rho[it]() = dot_prod(r_func[it - 1], r_func[it - 1], width, height, ch);

            //Compute p
            p[it] = Func("p_cg_" + itstr + "_" + oitstr  );
            if( it == 1 )
            {
                // p = r
                p[it](x, y, c) = r_func[it - 1](x, y, c);
            }
            else
            {
                // p = r + (rho / rho_1)*p;
                p[it](x, y, c) = r_func[it - 1](x, y, c) + ( rho[it]() / rho[it - 1]() ) * p[it - 1](x, y, c);
            }
            
            //Compute q = Ap
            //q = Mfun(p);
            Func AtA_p("AtA_p_cg_" + itstr + "_" + oitstr );
            AtA_p = AtA_conv(p[it], width, height, K_func, width_kernel, height_kernel, MtM_func);

            Func KtK_p("KtK_p_cg" + itstr + "_" + oitstr );
            KtK_p = KTK_grad_mat(p[it], width, height);

            Func q("q_cg_" + itstr + "_" + oitstr  );
            q(x, y, c) = AtA_p(x, y, c) + beta_cg * KtK_p(x, y, c);

            
            //Compute alpha
            //alpha = rho / (p(:)'*q(:) );
            Func pq("pq_cg_" + itstr + "_" + oitstr  );
            pq() = dot_prod(p[it], q, width, height, ch);
            
            Func alpha("alpha_cg_" + itstr  + "_" + oitstr );
            alpha() = rho[it]() / pq();
            
            //Compute x
            //x = x + alpha * p; % update approximation vector
            x_func[it] = Func("xout_cg_" + itstr + "_" + oitstr );
            x_func[it](x,y,c) = x_func[it - 1](x,y,c) + alpha() * p[it](x,y,c);
            
            //Compute r
            //r = r - alpha*q;  ; % compute residual
            r_func[it] = Func("rout_cg_" + itstr + "_" + oitstr );
            r_func[it](x,y,c) = r_func[it - 1](x,y,c) - alpha() * q(x,y,c);
            
            //Schedule r
            x_func[it].compute_root();
            r_func[it].compute_root();
        }

        //Compute the norm of r for final iterate
        Func norm_r("norm_r");
        norm_r() = norm_L2(r_func[cgiters], width, height, ch);

        std::vector<Func> outputs;
        outputs.push_back(x_func[cgiters]); 
        outputs.push_back(norm_r);     
        
        //Compute the norm of r for final iterate
        //return x_func[cgiters];
        return outputs;
}

#endif //PROX_OPERATORS_IMAGE_FORM_H