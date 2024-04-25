#ifndef MODEL_H
#define MODEL_H



# include "..\Model\Drift.h"
# include "..\Model\Noise.h"
# include "..\Model\Bound.h"
# include "..\Model\Bias.h"

class Model
{
public:
    // This four pointers must be removed to not leak memory.
    Drift* ptr_drift;
    Noise* ptr_noise;
    Bound* ptr_bound;
    Bias* ptr_bias;
    double dx, dt, max_time, drift0, noise0, b0, time_coefficient;

    // Constructor prototypes
    Model() {}
    // Test t_domain and x_domain
    Model(Bound* bound, double dx, double dt, double max_time) : ptr_bound(bound), dx(dx), dt(dt), max_time(max_time)
    {
        if (dx > .01)
        {
            std::cout << "Warning: dx > 0.01.\n";
        }
        if (dt > .01)
        {
            std::cout << "Warning: dt > 0.01 s.\n";
        }
        if (max_time > 10)
        {
            // std::cout << "Warning: tmax > 10 s.\n";
        }
    }

    // Test flux
    Model(Drift* drift, Noise* noise, Bound* bound, double dx, double dt, double max_time) : ptr_drift(drift), ptr_noise(noise), ptr_bound(bound), dx(dx), dt(dt), max_time(max_time)
    {
        if (dx > .01)
        {
            std::cout << "Warning: dx > 0.01.\n";
        }
        if (dt > .01)
        {
            std::cout << "Warning: dt > 0.01 s.\n";
        }
        if (max_time > 10)
        {
            // std::cout << "Warning: tmax > 10 s.\n";
        }
    }

    // Test get_starting_pdf
    Model(Drift* drift, Noise* noise, Bound* bound, Bias* bias, double dx, double dt, double max_time) : ptr_drift(drift), ptr_noise(noise),
        ptr_bound(bound), ptr_bias(bias), dx(dx), dt(dt), max_time(max_time)
    {
        if (dx > .01)
        {
            std::cout << "Warning: dx > 0.01.\n";
        }
        if (dt > .01)
        {
            std::cout << "Warning: dt > 0.01 s.\n";
        }
        if (max_time > 10)
        {
            // std::cout << "Warning: tmax > 10 s.\n";
        }
    }

    Model(Drift* drift, Noise* noise, Bound* bound, Bias* bias, double dx, double dt, double max_time,
        double drift0, double noise0, double b0) : ptr_drift(drift), ptr_noise(noise), ptr_bound(bound), ptr_bias(bias), dx(dx), dt(dt), max_time(max_time), drift0(drift0), noise0(noise0), b0(b0)
    {
        if (dx > .01)
        {
            std::cout << "Warning: dx > 0.01.\n";
        }
        if (dt > .01)
        {
            std::cout << "Warning: dt > 0.01 s.\n";
        }
        if (max_time > 10)
        {
            // std::cout << "Warning: tmax > 10 s.\n";
        }
    }
    virtual ~Model()
    {
        delete ptr_drift;
        delete ptr_noise;
        delete ptr_bound;
        delete ptr_bias;
    }

    arma::sp_mat splice(arma::sp_mat m, int lower, int upper)
    {
        arma::vec diag(m.diag());
        arma::vec up(m.diag(1));
        arma::vec down(m.diag(-1));
        unsigned ndiag = diag.n_elem;
        while (upper < 0)
        {
            upper += ndiag;
        }

        arma::sp_mat out = arma::sp_mat(upper - lower, upper - lower);
        out.diag() = diag.subvec(lower, upper - 1);
        out.diag(1) = up.subvec(lower, upper - 2);
        out.diag(-1) = down.subvec(lower, upper - 2);

        return out;
    }

    arma::vec t_domain()
    {
        return arma::regspace(0, dt, max_time + 0.1 * dt);
        // why 0.1 instead of 1, if want to include nax_time into last bin.
    }
    arma::vec x_domain()
    {
        std::string bound_name = ptr_bound->get_name();

        // t is None
        arma::vec time_vector = t_domain();
        arma::vec B_tmp(time_vector.n_elem);
        for (size_t i = 0; i < time_vector.n_elem; i++)
        {
            B_tmp[i] = ptr_bound->get_bound(time_vector[i]);
        }

        // Align the bound to dx borders
        double B = std::ceil(B_tmp.max() / dx) * dx;
        // 0.1 * dx is to ensure that the largest number in the array is B
        return arma::regspace(-B, dx, B + 0.1 * dx);
    }

    double flux(double x_bound, double t)
    {
        double drift_flux = ptr_drift->get_flux(x_bound, dx, dt);
        double noise_flux = ptr_noise->get_flux(x_bound, dx, dt);
        return drift_flux + noise_flux;
    }
    arma::vec get_starting_pdf()
    {
        // The evidence space at time step 0.
        //
        // Returns a N-element vector (where N is the size of x_domain())
        // which should sum to 1.
        return ptr_bias->get_bias(x_domain());
    }

    // Solving the SDE =================================================
    arma::field<arma::vec> solve_numerical_cn()
    {
        std::cout << "Solving the SDE using Crank-Nicolson scheme.\n";

        // Solve the model using Crank-Nicolson scheme.
        //
        // Crank-Nicolson scheme solves the model at each time point.
        // Results are then compiled together.  This is the core DDM solver of this
        // library. It returns a Solution object describing the joint PDF.
        arma::vec pdf_curr = get_starting_pdf(); // Initial condition
        arma::vec pdf_outer = get_starting_pdf();
        arma::vec pdf_inner = get_starting_pdf();

        arma::vec pdf_outer_prev = pdf_outer;
        arma::vec pdf_inner_prev = pdf_inner;

        // If pdf_corr + pdf_err + undecided probability are summed, they should
        // equal 1. So these are components of the joint pdf.
        arma::vec ts = t_domain();
        arma::vec xs = x_domain();
        unsigned nt = ts.n_elem;
        unsigned nx = xs.n_elem;
        arma::vec pdf_O = arma::zeros(nt);
        arma::vec pdf_X = arma::zeros(nt);

        double bound_shift = 0;
        // Note that we linearly approximate the bound by the two surrounding grids sandwiching it.
        unsigned x_index_inner = std::ceil(bound_shift / dx);  // Index for the inner bound (smaller matrix)
        unsigned x_index_outer = std::floor(bound_shift / dx); // Index for the outer bound (larger matrix)
        unsigned x_index_inner_prev, x_index_outer_prev;
        unsigned x_index_inner_shift, x_index_outer_shift;

        // We weight the upper and lower matrices according to how far away the bound
        // is from each.  The weight of each matrix is approximated linearly. The
        // inner bound is 0 when bound exactly at grids.
        double weight_inner = (bound_shift - x_index_outer * dx) / dx;
        double weight_outer = 1. - weight_inner; // The weight of the lower bound matrix, approximated linearly.
        double weight_inner_prev, weight_outer_prev;

        unsigned end = nx - x_index_outer - 1;
        arma::vec xs_inbounds = xs.subvec(x_index_outer, end); // List of x-positions still within bounds.
        arma::vec xs_inbounds_prev;

        // each step computes the value for the next time point
        double bound;
        double b0 = ptr_bound->get_bound(0); // boundary at the time = 0
        // Rcout << "b0: " << b0 << "\n";
        arma::sp_mat drift_matrix, noise_matrix, diffusion_matrix;
        arma::sp_mat drift_matrix_prev, noise_matrix_prev, diffusion_matrix_prev;

        for (size_t i = 0; i < (nt - 1); i++)
        {
            // Rcout << i << ": " << ts[i] << "\t";
            // Update Previous state.
            pdf_outer_prev = pdf_outer;
            pdf_inner_prev = pdf_inner;

            // For efficiency only do diffusion if there's at least some densities
            // remaining in the channel.
            if (arma::accu(pdf_curr) > 1e-4)
            {
                // Define the boundaries at current time.
                bound = ptr_bound->get_bound(ts[i]); // Boundary at current time-step.
                // Now figure out which x positions are still within the (collapsing) bound.
                // If the boundary at time = 0 is smaller than new bound, the bound is
                // expanded
                if (b0 < bound)
                    std::cout << "Changed bound " << bound << " baseline b0 " << b0 << std::endl;
                bound_shift = b0 - bound;
                // Linearly approximate the bound by the two surrounding grids sandwiching it.
                x_index_inner_prev = x_index_inner;
                x_index_outer_prev = x_index_outer;

                x_index_inner = std::ceil(bound_shift / dx);  // Index for the inner bound (smaller matrix)
                x_index_outer = std::floor(bound_shift / dx); // Index for the outer bound (larger matrix)

                x_index_inner_shift = x_index_inner - x_index_inner_prev;
                x_index_outer_shift = x_index_outer - x_index_outer_prev;
                unsigned x_index_io_shift = x_index_inner - x_index_outer;
                unsigned x_index_io_shift_prev = x_index_inner_prev - x_index_outer_prev;
                // We weight the upper and lower matrices according to how far away the
                // bound is from each.  The weight of each matrix is approximated
                // linearly. The inner bound is 0 when bound exactly at grids.
                weight_inner_prev = weight_inner;
                weight_outer_prev = weight_outer;
                weight_inner = (bound_shift - x_index_outer * dx) / dx; // 0
                weight_outer = 1. - weight_inner;                       // The weight of the lower bound matrix, approximated linearly. 1

                xs_inbounds_prev = xs_inbounds; // List of x-positions still within bounds.
                end = nx - x_index_outer - 1;
                xs_inbounds = xs.subvec(x_index_outer, end); // List of x-positions still within bounds.

                // Diffusion Matrix
                drift_matrix = ptr_drift->get_matrix(xs_inbounds, ts[i], dx, dt);
                drift_matrix *= .5; // Crank Nicolson scheme, theta = .5

                noise_matrix = ptr_noise->get_matrix(xs_inbounds, ts[i], dx, dt);
                noise_matrix *= .5;

                // x_lst_inbounds was changed within the loop, so we needed to update it
                // size in every time step.
                unsigned nx_inbounds = xs_inbounds.n_elem;
                diffusion_matrix = arma::speye<arma::sp_mat>(nx_inbounds, nx_inbounds);
                diffusion_matrix += drift_matrix;
                diffusion_matrix += noise_matrix;

                double t_tmp = std::max(0., ts[i] - dt);
                drift_matrix_prev = ptr_drift->get_matrix(xs_inbounds_prev, t_tmp, dt, dx);
                drift_matrix_prev *= 0.5;
                noise_matrix_prev = ptr_noise->get_matrix(xs_inbounds_prev, t_tmp, dt, dx);
                noise_matrix_prev *= 0.5;

                diffusion_matrix_prev = arma::speye<arma::sp_mat>(nx_inbounds, nx_inbounds);
                diffusion_matrix_prev -= drift_matrix_prev;
                diffusion_matrix_prev -= noise_matrix_prev;

                //====================================================================
                // Compute Probability density functions (pdf)
                //====================================================================
                // PDF for outer matrix
                // Considers the whole space in the previous step for matrix
                // multiplication, then restrains to current space when solving for
                // matrix_diffusion. Separating outer and inner pdf_prev
                //
                // For constant bounds pdf_inner is unnecessary.
                // For changing bounds pdf_inner is always needed,
                // even if the current inner and outer bounds coincide. I currently make
                // this generally but we can constrain it to changing-bound simulations
                // only.
                unsigned so_from = x_index_outer_shift;             // 0
                unsigned so_to = nx_inbounds + x_index_outer_shift; // 201
                unsigned si_from = x_index_io_shift;                // 0
                unsigned si_to = nx_inbounds - x_index_io_shift;    // 201

                unsigned si2_from = x_index_io_shift_prev;                         // 0
                unsigned si2_to = xs_inbounds_prev.n_elem - x_index_io_shift_prev; // 201
                unsigned si3_from = x_index_inner_shift;                           // 0
                unsigned si3_to = nx - 2 * x_index_inner + x_index_inner_shift;    // 201

                arma::vec term0 = diffusion_matrix_prev * pdf_outer_prev; // 201 x 201 * 201 x 1 = 201 x 1
                arma::vec term1 = term0.subvec(so_from, so_to - 1);       // 0 to 200 (201 elements)

                // 201; X = spsolve(A, B);  A * X = B, return X
                pdf_outer = arma::spsolve(diffusion_matrix, term1);

                // Should always be the case, since we removed CN changing bounds support
                if (x_index_inner == x_index_outer)
                {
                    // if (i == 0) Rcout <<" If was ran\n";
                    pdf_inner = pdf_outer;
                }
                else
                {
                    arma::sp_mat tmp_sp0 = splice(diffusion_matrix, si_from, si_to);
                    arma::sp_mat tmp_sp1 = splice(diffusion_matrix_prev, si2_from, si2_to);
                    arma::vec tmp_sp2 = tmp_sp1 * pdf_inner_prev;
                    arma::vec tmp_sp3 = tmp_sp2.subvec(si3_from, si3_to - 1);
                    pdf_inner = arma::spsolve(tmp_sp0, tmp_sp3);
                }

                // pdfs out of bound is considered decisions made.
                unsigned tmp_idx0 = pdf_outer_prev.n_elem - 1;
                unsigned tmp_idx1 = pdf_inner_prev.n_elem - 1;
                unsigned tmp_idx2 = pdf_outer_prev.n_elem - x_index_outer_shift;
                unsigned tmp_idx3 = pdf_inner_prev.n_elem - x_index_inner_shift;

                double tmp0, tmp1, tmp2, tmp3;
                if (x_index_outer_shift == 0)
                {
                    tmp0 = 0;
                    tmp2 = 0;
                }
                else
                {
                    tmp0 = weight_outer_prev * arma::accu(pdf_outer_prev.subvec(0, x_index_outer_shift - 1));
                    tmp2 = weight_outer_prev * arma::accu(pdf_outer_prev.subvec(tmp_idx2, tmp_idx0));
                }
                if (x_index_inner_shift == 0)
                {
                    tmp1 = 0;
                    tmp3 = 0;
                }
                else
                {
                    tmp1 = weight_inner_prev * arma::accu(pdf_inner_prev.subvec(0, x_index_inner_shift - 1));
                    tmp3 = weight_inner_prev * arma::accu(pdf_inner_prev.subvec(tmp_idx3, tmp_idx1));
                }

                pdf_X[i + 1] += tmp0 + tmp1;
                pdf_O[i + 1] += tmp2 + tmp3;
                // Reconstruct current probability density function, adding outer and
                // inner contribution to it.  Use .fill() method to avoid allocating
                // memory with np.zeros().
                pdf_curr.fill(0); // Reset
                pdf_curr.subvec(x_index_outer, nx - x_index_outer - 1) += weight_outer * pdf_outer;
                pdf_curr.subvec(x_index_inner, nx - x_index_inner - 1) += weight_inner * pdf_inner;
            }
            else
            {
                break;
            } // break if the remaining densities are too small....

            // Increase current, transient probability of crossing either bounds, as
            // flux.  Corr is a correct answer, err is an incorrect answer
            double _inner_B_corr = xs[nx - 1 - x_index_inner];
            double _outer_B_corr = xs[nx - 1 - x_index_outer];
            double _inner_B_err = xs[x_index_inner];
            double _outer_B_err = xs[x_index_outer];
            if (pdf_inner.n_elem == 0)
            { // Otherwise we get an error when bounds collapse to 0
                pdf_inner = arma::zeros(0);
            }

            double flux_outer_B_corr = flux(_outer_B_corr, ts[i]);
            double flux_inner_B_corr = flux(_inner_B_corr, ts[i]);
            double flux_outer_B_err = flux(_outer_B_err, ts[i]);
            double flux_inner_B_err = flux(_inner_B_err, ts[i]);

            pdf_O[i + 1] += 0.5 * weight_outer * pdf_outer[pdf_outer.n_elem - 1] * flux_outer_B_corr +
                0.5 * weight_inner * pdf_inner[pdf_inner.n_elem - 1] * flux_inner_B_corr;

            pdf_X[i + 1] += 0.5 * weight_outer * pdf_outer[0] * flux_outer_B_err +
                0.5 * weight_inner * pdf_inner[0] * flux_inner_B_err;

            pdf_O[i] += 0.5 * weight_outer * pdf_outer[pdf_outer.n_elem - 1] * flux_outer_B_corr +
                0.5 * weight_inner * pdf_inner[pdf_inner.n_elem - 1] * flux_inner_B_corr;

            pdf_X[i] += 0.5 * weight_outer * pdf_outer[0] * flux_outer_B_err +
                0.5 * weight_inner * pdf_inner[0] * flux_inner_B_err;

            // Renormalize when the channel size has <1 grid, although
            // all hell breaks loose in this regime.
            if (bound < dx)
            {
                pdf_O[i + 1] *= (1 + (1 - bound / dx));
                pdf_X[i + 1] *= (1 + (1 - bound / dx));
            }
        }

        // Detect and fix below zero errors.  Here, we don't worry about undecided
        // probability as we did with the implicit method, because CN tends to
        // oscillate around zero, especially when noise (sigma) is large.  The user
        // would be directed to decrease dt.
        arma::vec pdf_U = pdf_curr;
        double minval = std::min(pdf_O.min(), pdf_X.min());
        if (minval < 0)
        {
            arma::uvec query0 = find(pdf_O < 0);
            arma::uvec query1 = find(pdf_X < 0);
            double sum_negative_strength = arma::accu(pdf_O.elem(query0)) +
                arma::accu(pdf_X.elem(query1));
            // For small errors, don't bother alerting the user
            if (sum_negative_strength < -.01)
            {
                std::cout << "Warning: probability density included values less than zero "
                    << "(minimum=" << minval << ", total=" << sum_negative_strength << ").  "
                    << "Please decrease h and/or avoid extreme parameter values.\n";
            }
            pdf_O.elem(query0).zeros();
            pdf_X.elem(query1).zeros();
        }

        // Fix numerical errors
        double pdfsum = arma::accu(pdf_O) + arma::accu(pdf_X);
        if (pdfsum > 1)
        {
            // If it is only a small renormalization, don't bother alerting the user.
            if (pdfsum > 1.01)
            {
                std::cout << "Warning: renormalizing probability density from " << pdfsum << " to 1. Try decreasing dt. If that doesn't eliminate this warning, "
                    << "it may be due to extreme parameter values and/or bugs in your "
                    << "model speficiation.\n";
            }
            pdf_O /= pdfsum;
            pdf_X /= pdfsum;
        }

        if ((arma::accu(pdf_O) + arma::accu(pdf_X)) > 1)
        {
            // Correct floating point errors to always get prob <= 1
            pdf_O /= 1.00000000001;
            pdf_X /= 1.00000000001;
        }
        arma::field<arma::vec> out(3);
        out[0] = pdf_O;
        out[1] = pdf_X;
        out[2] = pdf_U;
        return out;
    }

    arma::field<arma::vec> solve_numerical(std::string method)
    {
        // Solve the PDE numerically.
        //
        // `method` = "explicit", "implicit", or "cn" (for Crank-Nicolson).
        //
        // Crank-Nicolson works for models with constant bounds, allegedly.
        // Implicit is the fall-back method. It should work well in most cases.
        //
        // Normally, the explicit method should not be used. Also note the stability
        // criteria for explicit method is:
        //
        //     | noise^2/2 * dt/dx^2 < 1/2
        //
        // Return: a two or three vectors field instance, for the PDF of each choices.
        //
        // return_evolution(default=False) governs whether the function
        // returns the full evolution of the pdf as part of the Solution object.
        // This only works with methods "explicit" or "implicit", not with "cn".
        if (method == "cn")
        {
            return solve_numerical_cn();
        }

        // std::cout << "Solving the SDE using the numerical method\n";

        // Initial condition of decision variable
        arma::vec pdf_curr = get_starting_pdf(); // Initial condition 201
        // Output correct and error pdfs.  If pdf_corr + pdf_err + undecided
        // probability are summed, they equal 1.  So these are components of the
        // joint pdf.
        arma::vec ts = t_domain();
        arma::vec xs = x_domain();
        unsigned nt = ts.n_elem;
        unsigned nx = xs.n_elem;
        arma::vec pdf_O = arma::zeros(nt);
        arma::vec pdf_X = arma::zeros(nt); // pdf_U is defined later

        // If evolution of pdf should be returned, preallocate np.array pdf_evolution
        // for performance reasons
        // mat pdf_evolution = arma::zeros<mat>(nx, nt);
        arma::mat pdf_evolution = arma::zeros<arma::mat>(nt, nx);

        // Find maximum bound for increasing bounds
        arma::vec bounds(nt); // note the plural
        for (size_t i = 0; i < nt; i++)
        {
            bounds[i] = ptr_bound->get_bound(ts[i]);
        }
        double bmax = bounds.max();
        // ================================================
        // Looping through time and updating the pdf.
        // ================================================
        arma::vec pdf_prev, xs_inbounds, pdf_outer, pdf_inner;
        double bound, bound_shift, weight_inner, weight_outer;
        unsigned x_index_inner, x_index_outer, nx_inbounds;
        arma::sp_mat drift_mat, noise_mat, diffusion_mat, diffusion_mat_explicit;

        for (size_t i = 0; i < (nt - 1); i++)
        {
            pdf_prev = pdf_curr; // Alias pdf_prev to pdf_curr for clarity
            // For efficiency only do diffusion if there's at least some densities
            // remaining in the channel.
            if (arma::accu(pdf_curr) < 1e-4)
                break; // Is 1e-4 reasonably small?

            // Boundary at current time-step.
            bound = ptr_bound->get_bound(ts[i]);
            // Now figure out which x positions are still within the (collapsing) bound.
            if (bmax < bound)
            {
                // Ensure the bound didn't expand
                std::cerr << "Warning: the bound expanded to " << bound << ", which is "
                    << "larger than maximum default bound " << bmax << ". Invalid change in bound.\n";
                exit(1);
            }

            // bmax is usually the bound at the time stamp 0, for collapsing bound scheme
            // Note that we linearly approximate the bound by the two surrounding grids
            // sandwiching it.
            bound_shift = bmax - bound;

            x_index_inner = std::ceil(bound_shift / dx);  // Index for the inner bound (smaller matrix)
            x_index_outer = std::floor(bound_shift / dx); // Index for the outer bound (larger matrix)
            // We weight the upper and lower matrices according to how far away the
            // bound is from each.  The weight of each matrix is approximated linearly
            // The inner bound is 0 when bound exactly at grids.
            weight_inner = (bound_shift - x_index_outer * dx) / dx;         // 0
            weight_outer = 1. - weight_inner;                               // The weight of the lower bound matrix, approximated linearly. 1
            xs_inbounds = xs.subvec(x_index_outer, nx - x_index_outer - 1); // List of x-positions still within bounds.
            nx_inbounds = xs_inbounds.n_elem;

            // Diffusion Matrix for Implicit method. Here defined as outer matrix,
            // and inner matrix is either trivial or an extracted submatrix.
            drift_mat = ptr_drift->get_matrix(xs_inbounds, ts[i], dx, dt);
            noise_mat = ptr_noise->get_matrix(xs_inbounds, ts[i], dx, dt);

            // if (i ==1 ) drift_mat.print("drift_mat");
            arma::vec term0 = pdf_prev.subvec(x_index_outer, nx - x_index_outer - 1);

            // =================================================================
            // Compute PDF for outer matrix
            if (method == "implicit")
            {

                // std::cout << "step " << i << " Using implicit method\n";

                try
                {
                    diffusion_mat = arma::speye<arma::sp_mat>(nx_inbounds, nx_inbounds) + drift_mat +
                        noise_mat;
                }
                catch (std::exception& e)
                {
                    std::cout << "speye + drift_mat + diffusion_mat error "
                        << "in 1st implicit method (numerical)\n";
                    std::cout << e.what();
                }
                try
                {
                    // X = spsolve(A, B);  A * X = B, return X
                    pdf_outer = arma::spsolve(diffusion_mat, term0);
                }
                catch (std::exception& e)
                {
                    // ptr_drift->parameters.t().print("\nAbnormal paraemters in arma::spsolve:");
                    // Rcpp::Rcout << "Resize pdf_outer to " << nx << " row\n";
                    pdf_outer = arma::zeros<arma::vec>(nx);
                    pdf_outer.fill(1e-10);
                    std::cout << e.what();
                }
            }
            else if (method == "explicit")
            {
                std::cout << "Using explicit method\n";

                // Explicit method flips sign except for the identity matrix
                diffusion_mat_explicit = arma::speye<arma::sp_mat>(nx_inbounds, nx_inbounds) -
                    drift_mat - noise_mat;
                // Compute PDF for outer matrix
                pdf_outer = diffusion_mat_explicit * term0; // (B.2)
            }
            else
            {
                std::cerr << "Unexpected method\n";
                exit(1);
            }
            // If the bounds are the same the bound perfectly aligns with the grid), we
            // don't need to solve the diffusion matrix again since we don't need a
            // linear approximation.
            arma::sp_mat tmp_sp0;
            arma::vec tmp_sp1;
            if (x_index_inner == x_index_outer)
            {
                pdf_inner = pdf_outer;
            }
            else
            {
                if (method == "implicit")
                {
                    tmp_sp0 = splice(diffusion_mat, 1, -1);
                    tmp_sp1 = pdf_prev.subvec(x_index_inner, nx - x_index_inner - 1);

                    try
                    {
                        // X = spsolve(A, B);  A * X = B, return X
                        pdf_inner = arma::spsolve(tmp_sp0, tmp_sp1);
                    }
                    catch (std::exception& e)
                    {
                        ptr_drift->parameters.t().print("\nAbnormal paraemters in arma::spsolve:");
                        std::cout << "Resize pdf_inner to " << nx << " row\n";
                        pdf_outer = arma::zeros<arma::vec>(nx);
                        pdf_outer.fill(1e-10);
                        std::cout << e.what();
                    }
                }
                else if (method == "explicit")
                {
                    tmp_sp0 = splice(diffusion_mat_explicit, 1, -1);
                    tmp_sp1 = pdf_prev.subvec(x_index_inner, nx - x_index_inner - 1);
                    pdf_inner = tmp_sp0 * tmp_sp1;
                }
                else
                {
                    std::cerr << "Unexpected method\n";
                    exit(1);
                }
            }
            // Pdfs out of bound is considered decisions made.
            double tmp0 = 0, tmp1 = 0, tmp2 = 0, tmp3 = 0;
            if (x_index_outer != 0)
            {
                tmp0 = arma::accu(pdf_prev.subvec(0, x_index_outer - 1));
                tmp1 = arma::accu(pdf_prev.subvec(0, x_index_inner - 1));
                tmp2 = arma::accu(pdf_prev.subvec(nx - x_index_outer, pdf_prev.n_elem - 1));
                tmp3 = arma::accu(pdf_prev.subvec(nx - x_index_inner, pdf_prev.n_elem - 1));
            }

            try
            {
                pdf_X[i + 1] += (weight_outer * tmp0 + weight_inner * tmp1);
                pdf_O[i + 1] += (weight_outer * tmp2 + weight_inner * tmp3);
            }
            catch (std::exception& e)
            {
                std::cout << " pdf_X[i+1] += error\n";
                std::cout << e.what();
            }

            // Reconstruct current probability density function, adding outer and inner
            // contribution to it.  Use .fill() method to avoid allocating memory with
            // np.zeros().
            pdf_curr.fill(0); // Reset

            // For explicit, should be pdf_outer.T?
            try
            {
                pdf_curr.subvec(x_index_outer, nx - x_index_outer - 1) += weight_outer * pdf_outer;
            }
            catch (std::exception& e)
            {
                ptr_drift->parameters.t().print("\nAbnormal paraemters in pdf_curr.subvec:");
                std::cout << " pdf_curr.subvec += weight_outer error\n";
                std::cout << "Resize pdf_outer to " << nx << " row\n";
                pdf_outer = arma::zeros<arma::vec>(nx);
                pdf_outer.fill(1e-10);
                std::cout << e.what();
            }

            try
            {
                pdf_curr.subvec(x_index_inner, nx - x_index_inner - 1) += weight_inner * pdf_inner;
            }
            catch (std::exception& e)
            {
                ptr_drift->parameters.t().print("\nAbnormal paraemters in pdf_curr.subvec:");
                std::cout << " pdf_curr.subvec += weight_inner error\n";
                std::cout << "Resize pdf_inner to " << nx << " row\n";
                pdf_inner = arma::zeros<arma::vec>(nx);
                pdf_inner.fill(1e-10);
                std::cout << e.what();
            }

            // Increase current, transient probability of crossing either bounds, as
            // flux. Corr is a correct answer, err is an incorrect answer
            double _inner_B_corr = xs[nx - 1 - x_index_inner];
            double _outer_B_corr = xs[nx - 1 - x_index_outer];
            double _inner_B_err = xs[x_index_inner];
            double _outer_B_err = xs[x_index_outer];
            if (pdf_inner.n_elem == 0)
            { // Otherwise we get an error when bounds collapse to 0
                pdf_inner = arma::zeros(0);
            }

            try
            {
                pdf_O[i + 1] += weight_outer * pdf_outer[pdf_outer.n_elem - 1] * flux(_outer_B_corr, ts[i]) +
                    weight_inner * pdf_inner[pdf_inner.n_elem - 1] * flux(_inner_B_corr, ts[i]);
                pdf_X[i + 1] += weight_outer * pdf_outer[0] * flux(_outer_B_err, ts[i]) +
                    weight_inner * pdf_inner[0] * flux(_inner_B_err, ts[i]);
            }
            catch (std::exception& e)
            {
                std::cout << " pdf_O or pdf_X += weight_outer error\n";
                std::cout << e.what();
            }

            // Renormalize when the channel size has <1 grid, although all hell breaks
            // loose in this regime.
            if (bound < dx)
            {
                pdf_O[i + 1] *= (1 + (1 - bound / dx));
                pdf_X[i + 1] *= (1 + (1 - bound / dx));
            }
        } // end of time loop

        // Detect and fix below zero errors
        arma::vec pdf_U = pdf_curr; // The evidence space at the last time point

        double min_tmp = std::min(pdf_O.min(), pdf_X.min());
        double minval = std::min(min_tmp, pdf_U.min());
        if (minval < 0)
        {
            arma::uvec query0 = find(pdf_O < 0);
            arma::uvec query1 = find(pdf_X < 0);
            arma::uvec query2 = find(pdf_U < 0);
            double sum_negative_strength = arma::accu(pdf_O.elem(query0)) + arma::accu(pdf_X.elem(query1));
            double sum_negative_strength_undec = arma::accu(pdf_U.elem(query2));

            if (sum_negative_strength < -.01)
            {
                // Rcout << "Warning: probability density included values less than zero " <<
                //   "(minimum = " << minval << ", total = " << sum_negative_strength <<
                //     "). Decrease the time step (h) may mitigate the problem to avoid " <<
                //       "extreme parameter values.\n";
                // Rcout << "(minimum = " << minval << ", total = " << sum_negative_strength
                // << ")\n";
            }
            if (sum_negative_strength_undec < -.01)
            {
                std::cout << "WARNING: remaining undecided densities included values less"
                    << " less than zero (minimum = " << minval << ", total="
                    << sum_negative_strength_undec << ").\n;";
            }
            // Setting these extreme densities to 0
            pdf_O.elem(query0).zeros();
            pdf_X.elem(query1).zeros();
            pdf_U.elem(query2).zeros();
        }

        // Fix numerical errors
        double pdfsum = arma::accu(pdf_O) + arma::accu(pdf_X) + arma::accu(pdf_U);

        if (pdfsum > 1)
        {
            if (pdfsum > 1.01)
            {
                // Rcout << "WARNING: renormalizing probability density from " << pdfsum <<
                //    " to 1.\n";
                // ptr_drift->parameters.t().print("\nAbnormal paraemters:");
            }

            try
            {
                pdf_O /= pdfsum;
            }
            catch (std::exception& e)
            {
                std::cout << " pdf_O /= pdfsum error\n";
                std::cout << e.what();
            }
            try
            {
                pdf_X /= pdfsum;
            }
            catch (std::exception& e)
            {
                std::cout << " pdf_X /= pdfsum error\n";
                std::cout << e.what();
            }
            try
            {
                pdf_U /= pdfsum;
            }
            catch (std::exception& e)
            {
                std::cout << " pdf_U /= pdfsum error\n";
                std::cout << e.what();
            }
        }

        // Correct floating point errors to always get prob <= 1
        if ((arma::accu(pdf_O) + arma::accu(pdf_X)) > 1)
        {
            try
            {
                pdf_O /= 1.00000000001;
            }
            catch (std::exception& e)
            {
                std::cout << " pdf_O /= error in solve_numerical\n";
                std::cout << e.what();
            }

            try
            {
                pdf_X /= 1.00000000001;
            }
            catch (std::exception& e)
            {
                std::cout << " pdf_X /= error in solve_numerical\n";
                std::cout << e.what();
            }
        }

        arma::field<arma::vec> out(3);
        out[0] = pdf_O;
        out[1] = pdf_X;
        out[2] = pdf_U;
        return out;
    }
};

#endif
#pragma once
