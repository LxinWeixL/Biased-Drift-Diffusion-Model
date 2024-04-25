#pragma once
#ifndef BOUND_H
#define BOUND_H

#include "..\Model\Dependence.h"
#include <iostream>
#include <armadillo>


class Bound : public Dependence
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
    double b0, time_coefficient;
    arma::vec condition, parameter;

    // Constant boundary constructor
    Bound(const std::string& name, const std::string& depname, double b0) : Dependence(name, depname), b0(b0)
    {
    }

    // Collapsing boundary constructor, version 1
    Bound(const std::string& name, const std::string& depname, double b0, double time_coefficient) : Dependence(name, depname), b0(b0), time_coefficient(time_coefficient) {}

    // Calculating the time-dependent boundary
    double get_bound(double t)
    {
        // t = one time point from a dicrete time vector
        double out;
        if (search_ign_case(name, "constant") != name.end())
        {
            out = b0;
        }
        else if (search_ign_case(name, "linear") != name.end())
        {
            // This cannot be solved by Crank-Nicolson solver?
            out = std::max(b0 - time_coefficient * t, 0.0);
        }
        else if (search_ign_case(name, "exponential") != name.end())
        {
            // This cannot be solved by Crank-Nicolson solver?
            out = b0 * std::exp(-time_coefficient * t);
        }
        else
        {
            std::throw_with_nested(std::runtime_error("The end of if-else loop in get_bound"));
        }
        return out;
    }

    virtual void print_name() const
    {
        std::cout << this->get_name() << " " << this->get_class_name() << "\n";
    }
    virtual ~Bound() {}
};

#endif
