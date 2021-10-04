#pragma once

#include <math.h>
#include <vector>

class Filter
{
private:
    float alpha = 0.97;

public:

    // Constructor/Destructor
    Filter();
    ~Filter();

    
    std::vector<double> emphasize_signal(std::vector<double> &signal);
    std::vector<double> hamming_window(std::vector<double> &signal);

};

Filter::Filter()
{
}

Filter::~Filter()
{
}

std::vector<double> Filter::emphasize_signal(std::vector<double> &signal)
{
    std::vector<double> emphasized;
    for(int i = 0; i < signal.size(); i++){
        if(i == 0)
        {
            emphasized.push_back(signal[i]);
        }
        else if(i != 0)
        {
            emphasized.push_back((signal[i] - (0.97 * signal[i-1])));
        }
    }
    return emphasized;
}

std::vector<double> Filter::hamming_window(std::vector<double> &signal)
{
    std::vector<double> window;

    for(int i = 0; i < signal.size(); ++i)
    {
        window.push_back(0.54 - 0.46 * cos(2*M_PI*i/signal.size()));
    }

    return window;
}