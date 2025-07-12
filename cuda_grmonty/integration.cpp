/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <tuple>

#include "cuda_grmonty/integration.hpp"

namespace integration {

// Only nodes in [0, 1], because of symmetry
static constexpr double xgk[31] = {0.9994844100504906, 0.9968934840746495, 0.9916309968704046, 0.9836681232797472,
                                   0.9731163225011263, 0.9600218649683075, 0.9443744447485598, 0.9262000474292743,
                                   0.9055733076999078, 0.8825605357920527, 0.8572052335460611, 0.8295657623827684,
                                   0.7997278358218391, 0.7677774321048262, 0.7337900624532268, 0.6978504947933158,
                                   0.6600610641266269, 0.6205261829892429, 0.5793452358263617, 0.5366241481420199,
                                   0.4924804678617786, 0.4470337695380892, 0.4004012548303944, 0.3527047255308781,
                                   0.3040732022736251, 0.2546369261678899, 0.2045251166823099, 0.1538699136085835,
                                   0.1028069379667370, 0.0514718425553177, 0.0000000000000000};

static constexpr double wgk[31] = {0.0013890136986770, 0.0038904611270999, 0.0066307039159313, 0.0092732796595178,
                                   0.0118230152534963, 0.0143697295070458, 0.0169208891890533, 0.0194141411939424,
                                   0.0218280358216092, 0.0241911620780806, 0.0265099548823331, 0.0287540487650413,
                                   0.0309072575623878, 0.0329814470574837, 0.0349793380280600, 0.0368823646518212,
                                   0.0386789456247276, 0.0403745389515359, 0.0419698102151642, 0.0434525397013560,
                                   0.0448148001331626, 0.0460592382710069, 0.0471855465692992, 0.0481858617570871,
                                   0.0490554345550299, 0.0497956834270742, 0.0504059214027823, 0.0508817958987496,
                                   0.0512215478492588, 0.0514261285374590, 0.0514947294294516};

// 15 Gauss weights (subset of wgk)
static constexpr double wg[15] = {0.0079681924961666, 0.0184664683110909, 0.0287847078833234, 0.0387991925696271,
                                  0.0484026728305941, 0.0574931562176191, 0.0659742298821805, 0.0737559747377052,
                                  0.0807558952294202, 0.0868997872010830, 0.0921225222377861, 0.0963687371746443,
                                  0.0995934205867953, 0.1017623897484055, 0.1028526528935588};

static std::tuple<double, double> qk61(const std::function<double(double)> &f, double a, double b);

double gauss_kronrod_61(const std::function<double(double)> &f, double a, double b, double eps_abs, double eps_rel,
                        int max_depth, int max_intervals) {
    int interval_count = 0;

    auto recurse = [&](auto &&self, double a, double b, int depth) -> double {
        if (interval_count >= max_intervals) {
            throw std::runtime_error("Exceeded maximum number of intervals.");
        }

        ++interval_count;

        auto [result, err] = qk61(f, a, b);
        const double tolerance = std::max(eps_abs, eps_rel * std::abs(result));

        if (err <= tolerance || depth >= max_depth) {
            return result;
        } else {
            const double mid = 0.5 * (a + b);
            return self(self, a, mid, depth + 1) + self(self, mid, b, depth + 1);
        }
    };

    return recurse(recurse, a, b, 0);
}

std::tuple<double, double> qk61(const std::function<double(double)> &f, double a, double b) {
    const double center = 0.5 * (a + b);
    const double half_length = 0.5 * (b - a);
    const double f_center = f(center);

    double result_kronrod = f_center * wgk[30];
    double result_gauss = 0.0;
    double resabs = std::abs(f_center) * wgk[30];
    double resasc = 0.0;

    for (int i = 0; i < 30; ++i) {
        const double absc = half_length * xgk[i];
        const double f1 = f(center - absc);
        const double f2 = f(center + absc);
        const double fsum = f1 + f2;

        result_kronrod += wgk[i] * fsum;
        resabs += wgk[i] * (std::abs(f1) + std::abs(f2));

        if (i % 2 == 1) { // Gauss nodes are subset of Kronrod (odd indices)
            result_gauss += wg[i / 2] * fsum;
        }
    }

    result_kronrod *= half_length;
    result_gauss *= half_length;
    resabs *= half_length;

    const double mean = result_kronrod / (b - a);

    resasc += wgk[30] * std::abs(f_center - mean);
    for (size_t i = 0; i < 30; ++i) {
        const double absc = half_length * xgk[i];
        const double f1 = f(center - absc);
        const double f2 = f(center + absc);
        resasc += wgk[i] * (std::abs(f1 - mean) + std::abs(f2 - mean));
    }
    resasc *= half_length;

    double err = std::abs(result_kronrod - result_gauss);

    if (resasc != 0.0 && err != 0.0) {
        double scale = std::pow((200 * err / resasc), 1.5);
        if (scale < 1.0) {
            err = resasc * scale;
        } else {
            err = resasc;
        }
    }

    return {result_kronrod, err};
}

}; /* namespace integration */
