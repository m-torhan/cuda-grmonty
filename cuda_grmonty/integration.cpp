/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>
#include <tuple>

#include "cuda_grmonty/integration.hpp"

namespace integration {

/* clang-format off */

/**
 * @brief Gauss-Kronrod nodes for 61-point integration.
 *
 * These are only the nodes in the interval [0, 1], leveraging symmetry to reduce the number of evaluations required.
 */
static constexpr double xgk[31] = {
  0.999484410050490637571325895705811,
  0.996893484074649540271630050918695,
  0.991630996870404594858628366109486,
  0.983668123279747209970032581605663,
  0.973116322501126268374693868423707,
  0.960021864968307512216871025581798,
  0.944374444748559979415831324037439,
  0.926200047429274325879324277080474,
  0.905573307699907798546522558925958,
  0.882560535792052681543116462530226,
  0.857205233546061098958658510658944,
  0.829565762382768397442898119732502,
  0.799727835821839083013668942322683,
  0.767777432104826194917977340974503,
  0.733790062453226804726171131369528,
  0.697850494793315796932292388026640,
  0.660061064126626961370053668149271,
  0.620526182989242861140477556431189,
  0.579345235826361691756024932172540,
  0.536624148142019899264169793311073,
  0.492480467861778574993693061207709,
  0.447033769538089176780609900322854,
  0.400401254830394392535476211542661,
  0.352704725530878113471037207089374,
  0.304073202273625077372677107199257,
  0.254636926167889846439805129817805,
  0.204525116682309891438957671002025,
  0.153869913608583546963794672743256,
  0.102806937966737030147096751318001,
  0.051471842555317695833025213166723,
  0.000000000000000000000000000000000
};

/**
 * @brief Gauss-Kronrod weights corresponding to xgk nodes.
 */
static constexpr double wgk[31] = {
  0.001389013698677007624551591226760,
  0.003890461127099884051267201844516,
  0.006630703915931292173319826369750,
  0.009273279659517763428441146892024,
  0.011823015253496341742232898853251,
  0.014369729507045804812451432443580,
  0.016920889189053272627572289420322,
  0.019414141193942381173408951050128,
  0.021828035821609192297167485738339,
  0.024191162078080601365686370725232,
  0.026509954882333101610601709335075,
  0.028754048765041292843978785354334,
  0.030907257562387762472884252943092,
  0.032981447057483726031814191016854,
  0.034979338028060024137499670731468,
  0.036882364651821229223911065617136,
  0.038678945624727592950348651532281,
  0.040374538951535959111995279752468,
  0.041969810215164246147147541285970,
  0.043452539701356069316831728117073,
  0.044814800133162663192355551616723,
  0.046059238271006988116271735559374,
  0.047185546569299153945261478181099,
  0.048185861757087129140779492298305,
  0.049055434555029778887528165367238,
  0.049795683427074206357811569379942,
  0.050405921402782346840893085653585,
  0.050881795898749606492297473049805,
  0.051221547849258772170656282604944,
  0.051426128537459025933862879215781,
  0.051494729429451567558340433647099
};

/**
 * @brief 15-point Gauss weights (subset of wgk) for faster integration when less precision is sufficient.
 */
static constexpr double wg[15] = {
  0.007968192496166605615465883474674,
  0.018466468311090959142302131912047,
  0.028784707883323369349719179611292,
  0.038799192569627049596801936446348,
  0.048402672830594052902938140422808,
  0.057493156217619066481721689402056,
  0.065974229882180495128128515115962,
  0.073755974737705206268243850022191,
  0.080755895229420215354694938460530,
  0.086899787201082979802387530715126,
  0.092122522237786128717632707087619,
  0.096368737174644259639468626351810,
  0.099593420586795267062780282103569,
  0.101762389748405504596428952168554,
  0.102852652893558840341285636705415
};

/* clang-format on */

/**
 * @brief Compute 61-point Gauss-Kronrod quadrature over [a, b] for a given function.
 *
 * @param f Function to integrate.
 * @param a Lower limit of integration.
 * @param b Upper limit of integration.
 *
 * @return Tuple containing (integral result, estimated error).
 */
static std::tuple<double, double> qk61(const std::function<double(double)> &f, double a, double b);

/**
 * @brief Represents an integration interval with result and error estimate.
 *
 * Useful for adaptive quadrature routines. The comparison operator is defined to create a max-heap based on the
 * largest error, allowing the algorithm to subdivide the interval with the largest estimated error first.
 */
struct Interval {
    double a, b;
    double result;
    double error;

    bool operator<(const Interval &other) const { return error < other.error; /* max-heap: largest error first */ }
};

double gauss_kronrod_61(
    const std::function<double(double)> &f, double a, double b, double eps_abs, double eps_rel, int max_intervals) {
    std::priority_queue<Interval> queue;

    auto [I, err] = qk61(f, a, b);
    queue.push({a, b, I, err});

    double total_result = I;
    double total_error = err;
    int intervals_used = 1;

    while (!queue.empty()) {
        if (total_error <= std::max(eps_abs, eps_rel * std::abs(total_result))) {
            break;
        }

        if (intervals_used >= max_intervals) {
            throw std::runtime_error("Failed to converge within max_intervals.");
        }

        Interval current = queue.top();
        queue.pop();

        double mid = 0.5 * (current.a + current.b);

        auto [I1, err1] = qk61(f, current.a, mid);
        auto [I2, err2] = qk61(f, mid, current.b);

        /* replace the current interval with its two halves */
        total_result += (I1 + I2 - current.result);
        total_error += (err1 + err2 - current.error);

        queue.push({current.a, mid, I1, err1});
        queue.push({mid, current.b, I2, err2});
        intervals_used += 1;
    }

    return total_result;
}

std::tuple<double, double> qk61(const std::function<double(double)> &f, double a, double b) {
    constexpr double eps = std::numeric_limits<double>::epsilon();

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

        if (i % 2 == 1) { /* Gauss nodes are subset of Kronrod (odd indices) */
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
        double scale = std::pow(200.0 * err / resasc, 1.5);
        err = (scale < 1.0) ? resasc * scale : resasc;
    }
    if (resasc == 0.0 || err < 50 * eps * resabs) {
        err = 0.0;
    }

    return {result_kronrod, err};
}

}; /* namespace integration */
