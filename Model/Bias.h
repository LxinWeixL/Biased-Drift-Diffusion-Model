#pragma once
#ifndef BIAS_H
#define BIAS_H

#include "..\Model\Dependence.h"
#include <iostream>
#include <armadillo>
#include <boost/math/distributions/normal.hpp>

class Bias : public Dependence
{
    // Bound dependence: bound collapses linearly over time.
    // - `m_b` - the bound at time t = 0.
    // - `m_tcoef` - (linear) the slope, i.e. the coefficient of time, should be
    //               greater than 0.
    //             - (exponential) one divided by the time constant for the
    //               collapse, should be greater than zero for collapsing bounds,
    //               less than zero for increasing bounds.  0 gives constant
    //               bounds.
private:
    void print_warning() const
    {
        std::cout << "The implementation assumes symmetrical boundaries. "
            << "For instance, when you enter 1, the boundaries are set at -1 and 1.\n";
        // stop("Please only enter a positive value");
    }

    // compare two characters, ignoring case to see if they are the same
    bool static comp_ign_case(char x, char y) { return tolower(x) == tolower(y); }

    // search for a substring in a string, ignoring case
    std::string::iterator search_ign_case(std::string& str, const std::string& substr)
    {
        return search(str.begin(), str.end(), substr.begin(), substr.end(), comp_ign_case);
    }

public:
    double x0, dx; // arbitrary starting point

    // Use the mid-point of the negative and the positive boundaries as the starting point
    Bias(const std::string& name, const std::string& depname) : Dependence(name, depname)
    {
    }

    // point and range x0 serves as sz when using either a uniform or a Gaussian distribution
    // to sample the initial starting point
    Bias(const std::string& name, const std::string& depname, double x0, double dx) : Dependence(name, depname), x0(x0), dx(dx)
    {
    }

    // Calculating a user defined initial evidence space. This will be a space vector.
    arma::vec get_bias(arma::vec xs)
    {
        // xs = the discrete point of the evidence space
        unsigned int nx = xs.n_elem;
        arma::vec out(nx);
        out.fill(0.0);

        if (search_ign_case(name, "centre") != name.end())
        {
            // This is an unbiased initial evidence space. The midpoint is x = 0.
            unsigned int mid_position = 0.5 * (xs.n_elem - 1);
            // At the time step 0, all densities are concentrated at the midpoint.
            // Drift and diffusion matrices will change the density along the evidence space. 
            out[mid_position] = 1.0; // Initial evidence space, assuming the midpoint is x = 0.
        }
        else if (search_ign_case(name, "point") != name.end())
        {
            // std::cout << "First bias assumption. The user can decide to place x0 any plance 
            // on the line of -B to B.";
            unsigned int start = round(x0 / dx);
            unsigned int shift_i = (double)start + 0.5 * (nx - 1);
            if ((shift_i < 0) | (shift_i >= nx))
            {
                std::throw_with_nested(std::runtime_error("An invalid initial condition"));
            }
            // Biased towards a point defined by x0. The user can define the point. If it closes to positive 
            // bound, the accumulator will reach there earlier, subject to the influence of the variability.
            //std::cout << "shift_i " << shift_i << " nx " << nx << "\n";
            out[shift_i] = 1.0;
        }
        else if (search_ign_case(name, "uniform") != name.end())
        {
            // Second bias assumption. The initial evidence space is a uniform distribution.
            out.fill(1.0 / nx);
        }
        else if (search_ign_case(name, "range") != name.end())
        {
            unsigned int shift_i = .5 * (nx - 1);
            unsigned sz_shift = x0 / dx; // x0 serves as m_sz
            unsigned from = shift_i - sz_shift;
            unsigned to = shift_i + sz_shift;
            out.subvec(from, to).fill(1.0);
            out = out / arma::accu(out);
        }
        else if (search_ign_case(name, "gaussian") != name.end())
        {
            // x0 serves as the standard deviation of the Gaussian distribution
            boost::math::normal_distribution<> norm(0, x0);
            for (size_t i = 0; i < xs.n_elem; i++)
            {
                out[i] = boost::math::pdf(norm, xs[i]);
            }
            out /= arma::accu(out);
        }
        else
        {
            std::throw_with_nested(std::runtime_error("The end of if-else loop in get_bias"));
        }

        return out;
    }


    virtual void print_name() const
    {
        std::cout << this->get_name() << " " << this->get_class_name() << "\n";
    }
    virtual ~Bias() {}
};

#endif
