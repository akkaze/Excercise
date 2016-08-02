#pragma once
#include <exception>
#include <string>

namespace cnn {
    class cnnError : public std::exception
    {
    public:
        explicit cnnError(const std::String& msg) : msg_(msg) {}
        const char* what() const throw() override { return msg_.c_str(); }

    private:
        std::string msg_;
    }
}