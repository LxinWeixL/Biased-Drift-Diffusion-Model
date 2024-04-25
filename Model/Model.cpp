


#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <iostream>
#include <string>
#include <cctype> // tolower
#include <armadillo>
#include <boost/math/distributions/normal.hpp>

#include "..\Model\Dependence.h"
#include "..\Model\Drift.h"
#include "..\Model\Noise.h"
#include "..\Model\Bound.h"
#include "..\Model\Bias.h"


// TODO: #include "Non_decision_time.hpp"
#include "Model.hpp"

#include <Python.h>

#include <numpy/arrayobject.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Python helper functions =================================================
py::array_t<double> armadilloToNumpy(const arma::mat& mat)
{
    // Get dimensions of the matrix
    ssize_t rows = mat.n_rows;
    ssize_t cols = mat.n_cols;

    // Create a Pybind11 array with the same shape as the Armadillo matrix
    py::array_t<double> result({ rows, cols });

    // Get a pointer to the data in the Armadillo matrix
    const double* data = mat.memptr();

    // Get a pointer to the data in the Pybind11 array
    double* result_data = result.mutable_data();

    // Copy data from Armadillo matrix to Pybind11 array
    std::memcpy(result_data, data, sizeof(double) * rows * cols);

    return result;
}

// Essential utilities =================================================
py::array_t<double> test_t_domain(double b0, double dx, double dt, double max_time)
{
    Bound* p_bound;
    Model* p_model;
    p_bound = new Bound("constant", "Bound", b0);
    p_model = new Model(p_bound, dx, dt, max_time);
    arma::vec t_domain = p_model->t_domain();
    return armadilloToNumpy(t_domain);
}

py::array_t<double> test_x_domain(double b0, double dx, double dt, double max_time)
{
    Bound* p_bound;
    Model* p_model;
    p_bound = new Bound("constant", "Bound", b0);
    p_model = new Model(p_bound, dx, dt, max_time);
    arma::vec x_domain = p_model->x_domain();
    return armadilloToNumpy(x_domain);
}

py::array_t<double> test_flux()
{
    double v0 = 0.05;
    double b0 = 1.0;
    double s0 = 0.1;

    double dx = 0.1;
    double dt = 0.1;
    double max_time = 3.0;
    Drift* p_drift;
    Noise* p_noise;
    Bound* p_bound;
    Model* p_model;
    p_drift = new Drift("constant", "Drift", v0);
    p_noise = new Noise("constant", "Noise", s0);
    p_bound = new Bound("constant", "Bound", b0);

    p_model = new Model(p_drift, p_noise, p_bound, dx, dt, max_time);
    arma::vec time_vector = p_model->t_domain();
    arma::vec flux_vector(time_vector.n_elem);

    for (size_t i = 0; i < time_vector.n_elem; i++)
    {
        double flux_i = p_model->flux(b0, time_vector[i]);
        flux_vector[i] = flux_i;
    }
    return armadilloToNumpy(flux_vector);
}


// res = Model.test_get_starting_pdf(bias, drift0, bound0, noise0, x0, dx, dt, max_time)

py::array_t<double> test_get_starting_pdf(std::string bias, double v0, double b0, double s0, double x0,
    double dx, double dt, double max_time)
{
    Drift* p_drift;
    Noise* p_noise;
    Bound* p_bound;
    Bias* p_bias;
    Model* p_model;
    p_drift = new Drift("constant", "Drift", v0);
    p_noise = new Noise("constant", "Noise", s0);
    p_bound = new Bound("constant", "Bound", b0);
    p_bias = new Bias(bias, "Bias", x0, dx);

    p_model = new Model(p_drift, p_noise, p_bound, p_bias, dx, dt, max_time);
    arma::vec starting_pdf = p_model->get_starting_pdf();
    return armadilloToNumpy(starting_pdf);
}

std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>> get_vectors()
{
    arma::vec v1 = arma::vec({ 1.0, 2.0, 3.0 });
    arma::vec v2 = arma::vec({ 4.0, 5.0, 6.0 });
    arma::vec v3 = arma::vec({ 7.0, 8.0, 9.0 });
    return std::make_tuple(py::array_t<double>(v1.n_elem, v1.memptr()),
        py::array_t<double>(v2.n_elem, v2.memptr()),
        py::array_t<double>(v3.n_elem, v3.memptr()));
}

std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>> test_solve_numerical_cn(
    double drift0, double bound0, double noise0, double dx, double dt, double max_time)
{
    Drift* p_drift = new Drift("constant", "Drift", drift0);
    Noise* p_noise = new Noise("constant", "Noise", noise0);
    Bound* p_bound = new Bound("constant", "Bound", bound0);
    Bias* p_bias = new Bias("centre", "Bias");

    Model* p_model = new Model(p_drift, p_noise, p_bound, p_bias, dx, dt, max_time);
    arma::field<arma::vec> out_field = p_model->solve_numerical_cn();
    return std::make_tuple(py::array_t<double>(out_field[0].n_elem, out_field[0].memptr()),
        py::array_t<double>(out_field[1].n_elem, out_field[1].memptr()),
        py::array_t<double>(out_field[2].n_elem, out_field[2].memptr()));
}

std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>> test_solve_numerical(
    std::string method, std::string bias, double drift0, double bound0, double noise0, double x0, double dx, double dt, double max_time)
{
    if (method == "centre") {
        throw std::invalid_argument("method must be one of 'cn', 'implicit', 'explicit'");
    }
    else if (method == "point") {
        throw std::invalid_argument("method must be one of ''cn', 'implicit', 'explicit'");
    }
    else {
    }


    // std::cout << "bias string " << bias << "\n";

    Drift* p_drift = new Drift("constant", "Drift", drift0);
    Noise* p_noise = new Noise("constant", "Noise", noise0);
    Bound* p_bound = new Bound("constant", "Bound", bound0);

    Bias* p_bias = new Bias(bias, "Bias", x0, dx);
    // p_bias->print_name();

    Model* p_model = new Model(p_drift, p_noise, p_bound, p_bias, dx, dt, max_time);
    arma::field<arma::vec> out_field = p_model->solve_numerical(method);
    return std::make_tuple(py::array_t<double>(out_field[0].n_elem, out_field[0].memptr()),
        py::array_t<double>(out_field[1].n_elem, out_field[1].memptr()),
        py::array_t<double>(out_field[2].n_elem, out_field[2].memptr()));
}

PYBIND11_MODULE(Model, m)
{
    m.doc() = "Model module";
    m.def("test_t_domain", &test_t_domain, "Test t_domain");
    m.def("test_x_domain", &test_x_domain, "Test x_domain");
    m.def("test_flux", &test_flux, "Test flux");
    m.def("test_get_starting_pdf", &test_get_starting_pdf, "Test get_starting_pdf");

    m.def("get_vectors", &get_vectors);
    m.def("test_solve_numerical_cn", &test_solve_numerical_cn, "Test solve_numerical_cn");
    m.def("test_solve_numerical", &test_solve_numerical, "Test solve_numerical");
}



