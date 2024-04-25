#pragma once
#include <cctype>

#include <iostream>
#include <armadillo>

class Dependence
{
protected:
    std::string name, class_name;

public:
    Dependence(const std::string& name,
        const std::string& class_name) :
        name(name), class_name(class_name) { }

    virtual const std::string& get_name() const { return name; }
    virtual const std::string& get_class_name() const { return class_name; }
    virtual ~Dependence()
    {

    }
    virtual void print_name() const = 0;  // pure virtual function
};

