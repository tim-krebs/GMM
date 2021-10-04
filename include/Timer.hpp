#pragma once

#include <chrono>
#include <atomic>

typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::nanoseconds    ns;
typedef std::chrono::microseconds   ms;
typedef std::chrono::milliseconds   mms;
typedef std::chrono::seconds        s;

class Timer{
private:
    std::chrono::time_point<Clock> t_start_point;

public:
    Timer() : t_start_point(Clock::now()) {}
    ~Timer() {}

    template <typename Units = ns, typename Rep = double>
    Rep elapsed_time() const
    {
        std::atomic_thread_fence(std::memory_order_relaxed);
        auto counted_time = std::chrono::duration_cast<Units>(Clock::now() - t_start_point).count();

        std::atomic_thread_fence(std::memory_order_relaxed);
        return static_cast<Rep>(counted_time);
    }

    void reset()
    {
        t_start_point = Clock::now();
    }

};