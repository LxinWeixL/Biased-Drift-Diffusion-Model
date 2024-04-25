#pragma once
#ifndef NOISE_H
#define NOISE_H

#include <iostream>
#include <armadillo>
#include "..\Model\Dependence.h"

class Noise : public Dependence
{
private:
    void print_warning() const
    {
        std::cout << "The implementation assumes symmetrical boundaries. "
            << "For instance, when you enter 1, the boundaries are set at -1 and 1.\n";
    }

    // compare two characters, ignoring case to see if they are the same
    bool static comp_ign_case(char x, char y) { return tolower(x) == tolower(y); }

    // search for a substring in a string, ignoring case
    std::string::iterator search_ign_case(std::string& str, const std::string& substr)
    {
        return search(str.begin(), str.end(), substr.begin(), substr.end(), comp_ign_case);
    }

public:
    double noise0, x_coefficient, t_coefficient;

    // Constant noise
    Noise(const std::string& name, const std::string& depname, double noise0) : Dependence(name, depname), noise0(noise0)
    {
        if (noise0 < 0)
            std::throw_with_nested(std::runtime_error("The noise level must be greater than 0"));
    }

    // Linear noise
    Noise(const std::string& name, const std::string& depname, double noise0, double t_coefficient) : Dependence(name, depname), noise0(noise0), t_coefficient(t_coefficient)
    {
        if (noise0 < 0)
            std::throw_with_nested(std::runtime_error("The noise level must be greater than 0"));
    }

    double get_noise(double x, double t)
    {
        double out;
        if (search_ign_case(name, "constant") != name.end())
        {
            out = noise0;
        }
        else if (search_ign_case(name, "linear") != name.end())
        {
            // The noise level decreases over time, because more evidence is accumulated.
            out = noise0 - t_coefficient * t;
            if (out < 0)
            {
                std::throw_with_nested(std::runtime_error("You entered a noise0 and a time cofficient resulting in negative noise"));
            }
        }
        else
        {
            std::throw_with_nested(std::runtime_error("The end of if-else loop in get_drift"));
        }
        return out;
    }

    double get_flux(double x_bound, double dx, double dt)
    {
        // t is a discrete time point
        double out;
        if (search_ign_case(name, "constant") != name.end())
        {
            out = 0.5 * dt / (dx * dx) * (noise0 * noise0);
        }
        else
        {
            std::throw_with_nested(std::runtime_error("The noise level must be greater than 0"));
        }
        return out;
    }

    arma::sp_mat get_matrix(arma::vec x, double t, double dx, double dt)
    {
        unsigned int nx = x.n_elem - 1; // nx = N = 400
        double dx2 = dx * dx;
        arma::vec up, down, diag;

        if (search_ign_case(name, "constant") != name.end())
        {
            // never use t here
            double noise2 = noise0 * noise0;
            diag.set_size(nx + 1);
            diag.fill(noise2 * dt / dx2);
            up.set_size(nx);
            up.fill(-.5 * noise2 * dt / dx2);
            down.set_size(nx);
            down.fill(-.5 * noise2 * dt / dx2);
        }
        else if (search_ign_case(name, "linear") != name.end())
        {
            // TODO fix this when it comes needed
            // arma::vec noise_t(x.n_elem);
            // for (size_t i = 0; i < xs.n_elem; i++)
            // {
            //   noise_t[i] = get_noise(x[i], t);
            // }
            // arma::vec term0 = 0.5 * (noise_t.subvec(1, nx) + noise_t.subvec(0, nx - 1));
            // diag = arma::square(noise_t) * dt / dx2;
            // up = -.5 * arma::square(term0) * dt / dx2;
            // down = -.5 * arma::square(term0) * dt / dx2;
            // out = TriDiagMatrix(diag, up, down);
            std::throw_with_nested(std::runtime_error("Non-constant noise not implemented"));
        }
        else
        {
            std::throw_with_nested(std::runtime_error("The end of if-else loop in get_drift"));
        }

        // 0:nx, 0:(nx-1), 1:nx
        // 0:nx, 1:nx, 0:(nx-1)
        arma::uvec diag_row = arma::regspace<arma::uvec>(0, 1, nx);
        arma::uvec up_row = arma::regspace<arma::uvec>(0, 1, nx - 1); // also be down col
        arma::uvec down_row = arma::regspace<arma::uvec>(1, 1, nx);   // also be a up col
        arma::uvec row0 = arma::join_cols(diag_row, up_row, down_row);
        arma::uvec row1 = arma::join_cols(diag_row, down_row, up_row);

        arma::umat locations = arma::join_rows(row0, row1).t();
        arma::vec values = arma::join_cols(diag, up, down);
        arma::sp_mat out(locations, values);
        return out;
    }

    virtual void print_name() const
    {
        std::cout << this->get_name() << " " << this->get_class_name() << "\n";
    }
    virtual ~Noise() {}
};

#endif
