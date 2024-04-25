#ifndef DRIFT_H
#define DRIFT_H


#include "..\Model\Dependence.h"

class Drift : public Dependence
{
private:
    // A sign function
    template <typename T>
    int sgn(T val) { return (T(0) < val) - (val < T(0)); }

    // compare two characters, ignoring case to see if they are the same
    bool static comp_ign_case(char x, char y) { return tolower(x) == tolower(y); }

    // search for a substring in a string, ignoring case
    std::string::iterator search_ign_case(std::string& str, const std::string& substr)
    {
        return search(str.begin(), str.end(), substr.begin(), substr.end(), comp_ign_case);
    }

public:
    double drift0;
    arma::vec parameters;
    arma::vec condition; // 2-element vector; first = D; second = TTA

    // constant
    Drift(const std::string& name, const std::string& depname, double drift0) : Dependence(name, depname), drift0(drift0)
    {
    }

    // Drift rate associated with a drift function
    Drift(const std::string& name,
        const std::string& depname,
        const double& drift0,
        const arma::vec& parameters,
        const arma::vec& condition) : Dependence(name, depname), drift0(drift0), parameters(parameters), condition(condition)
    {
    }

    double get_value(double t);

    double get_drift(double x, double t)
    {
        double out;
        if (search_ign_case(name, "constant") != name.end())
        {
            // if the name contains "constant", then it is a constant drift rate
            out = drift0;
        }
        else if (search_ign_case(name, "linear") != name.end())
        {
            std::throw_with_nested(std::runtime_error("Linear drift rate is not implemented yet."));
        }
        else
        {
            std::throw_with_nested(std::runtime_error("The end of if-else loop in get_drift"));
        }
        return out;
    }

    // For constant drift rate.
    double get_flux(double x_bound, double dx, double dt)
    {
        // x_bound is the boundary of the evidence state
        // t is the time
        // dx is the grid size
        // dt is the time step
        double out;
        if (search_ign_case(name, "constant") != name.end())
        {
            out = .5 * dt / dx * sgn(x_bound) * drift0;
        }
        else
        {
            std::throw_with_nested(std::runtime_error("The end of if-else loop in get_flux"));
        }
        return out;
    }

    // Other complex drift rates
    arma::vec get_flux(double x_bound, double t, double dx, double dt, double alpha,
        double beta, double gamma)
    {
        arma::vec out;
        if (search_ign_case(name, "linear") != name.end())
        {
            out = .5 * dt / dx * sgn(x_bound) * get_drift(x_bound, t);
        }
        else
        {
            std::throw_with_nested(std::runtime_error("The end of if-else loop in get_flux"));
        }
        return out;
    }

    arma::sp_mat get_matrix(arma::vec x, double t, double dx, double dt)
    {
        // x is the vector of evidence state
        // t is a time point. It becomes essential whne the drift rate is not constant.  
        // dx is the grid size
        // dt is the time step

        // note the minus 1 here, bcz we want only the up and down, which has one
        // element less than the nrow and ncol of the matrix of evidence state,
        // which is a square matrix.
        unsigned nx = x.n_elem - 1;
        arma::sp_mat out = arma::sp_mat(nx + 1, nx + 1);
        arma::vec up, down;

        if (search_ign_case(name, "constant") != name.end())
        {
            up.set_size(nx); // xs does not play any role in the case of constant
            down.set_size(nx);
            up.fill(.5 * dt / dx * drift0);
            down.fill(-.5 * dt / dx * drift0);
        }
        else if (search_ign_case(name, "linear") != name.end())
        {

            arma::vec rates(x.n_elem);
            for (size_t i = 0; i < x.n_elem; i++)
                rates[i] = get_drift(x[i], t);
            up = .5 * dt / dx * rates.subvec(1, nx);        // 400
            down = -.5 * dt / dx * rates.subvec(0, nx - 1); // 400
        }
        else
        {
            std::throw_with_nested(std::runtime_error("The end of if-else loop in get_flux"));
        }

        out.diag(1) = up; // Use Armadillo black magic
        out.diag(-1) = down;

        return out;
    }

    virtual void print_name() const
    {
        std::cout << this->get_name() << " " << this->get_class_name() << "\n";
    }

    virtual ~Drift() {}
};

#endif
#pragma once
