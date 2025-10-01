/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math_constants.h>

#include <cstdio>
#include <queue>
#include <semaphore>
#include <tuple>

#include "cuda_grmonty/consts.hpp"
#include "cuda_grmonty/harm_data.cuh"
#include "cuda_grmonty/harm_model.cuh"
#include "cuda_grmonty/photon.cuh"
#include "cuda_grmonty/proba.cuh"
#include "cuda_grmonty/radiation.cuh"
#include "cuda_grmonty/super_photon.cuh"
#include "cuda_grmonty/utils.cuh"

#include "cuda_grmonty/harm_data.hpp"
#include "cuda_grmonty/photon.hpp"
#include "cuda_grmonty/utils.hpp"

namespace cuda_super_photon {

const int n_photons = consts::cuda::n_photons;

/**
 * @brief Maximum scattering optical depth per photon (device).
 */
__device__ double dev_max_tau_scatt = 0.0;

/**
 * @brief Number of super-photons recorded (device).
 */
__device__ int dev_n_super_photon_recorded = 0;

/**
 * @brief Number of super-photons that scattered (device).
 */
__device__ int dev_n_super_photon_scatt = 0;

/**
 * @brief Device pointer to simulation header.
 */
static struct harm::Header *dev_header;

/**
 * @brief Device copy of simulation data (geometry, fluid, zones).
 */
static struct cuda_harm::Data dev_data;

/**
 * @brief Device pointer to units structure.
 */
static struct harm::Units *dev_units;

/**
 * @brief Device copy of precomputed tables for GPU calculations.
 */
static struct cuda_harm::Tables dev_tables;

/**
 * @brief Device pointer to photon spectrum accumulator.
 */
static struct harm::Spectrum *dev_spectrum;

/**
 * @enum PhotonState
 * @brief Represents the state of a photon in the simulation.
 */
enum PhotonState : uint8_t {
    Empty = 0,       /* Photon slot is empty. */
    New = 1,         /* Newly created photon, not yet initialized. */
    Initialized = 2, /* Photon initialized with position and momentum. */
    Tracked = 3      /* Photon has been propagated/tracked. */
};

/**
 * @brief Initialize CUDA random number generator states for super-photon propagation.
 *
 * Each thread initializes its own state for use in Monte Carlo sampling.
 *
 * @param rng_state Pointer to device array of random number generator states.
 */
static __global__ void init_rng(curandStatePhilox4_32_10_t *rng_state);

/**
 * @brief Load new photons and validate them.
 *
 * Checks if photons are in a valid state and prepares new photons for initialization.
 *
 * @param photon       Device array of existing photons.
 * @param photon_new   Device array for newly created photons.
 * @param photon_state Device array of photon states to track initialization and tracking.
 */
static __global__ void
load_validate_photon(struct PhotonArray photon, struct PhotonArray photon_new, enum PhotonState *photon_state);

/**
 * @brief Setup per-photon propagation variables before starting photon tracking.
 *
 * Computes initial local fluid properties, photon frequency, absorption, and scattering opacities, as well as bias
 * factors for weighted Monte Carlo propagation.
 *
 * @param header       Pointer to simulation header (grid, units, parameters).
 * @param data         Device copy of simulation data (geometry, fluid, zones).
 * @param units        Pointer to unit conversion structure.
 * @param tables       Precomputed tables (hotcross, k2, f) on device memory.
 * @param bias_norm    Bias normalization factor for photon weighting.
 * @param photon       Device array of photons to initialize.
 * @param photon_state Device array of photon states.
 * @param n_step       Device array for photon step counters.
 * @param fluid_n_e    Device array of local electron number densities.
 * @param theta        Device array of local electron temperatures (theta_e).
 * @param nu           Device array of photon frequencies in the fluid frame.
 * @param alpha_scatti Device array of inverse scattering opacities.
 * @param alpha_absi   Device array of inverse absorption opacities.
 * @param bi           Device array of photon bias factors.
 */
static __global__ void setup_variables(const struct harm::Header *header,
                                       const struct cuda_harm::Data data,
                                       const struct harm::Units *units,
                                       const struct cuda_harm::Tables tables,
                                       double bias_norm,
                                       struct PhotonArray photon,
                                       enum PhotonState *photon_state,
                                       int *n_step,
                                       double *fluid_n_e,
                                       double *theta,
                                       double *nu,
                                       double *alpha_scatti,
                                       double *alpha_absi,
                                       double *bi);

/**
 * @brief Apply stopping criterion to photons during propagation.
 *
 * Determines whether each photon should stop propagating based on optical depth, boundary conditions, or other
 * termination criteria.
 *
 * @param rng_state    Device array of random number generator states for stochastic checks.
 * @param header       Pointer to simulation header (grid, units, parameters).
 * @param photon       Device array of photons to check.
 * @param photon_state Device array of photon states, updated if photon should stop.
 */
static __global__ void stop_criterion(curandStatePhilox4_32_10_t *rng_state,
                                      const struct harm::Header *header,
                                      struct PhotonArray photon,
                                      enum PhotonState *photon_state);

/**
 * @brief Compute the propagation step size for each photon.
 *
 * Determines the distance each photon should move in this iteration based on local fluid properties, optical depths,
 * and geometry.
 *
 * @param header       Pointer to simulation header.
 * @param photon       Device array of photons to compute step sizes for.
 * @param photon_state Device array of photon states.
 * @param step_size    Device array to store computed step sizes.
 */
static __global__ void step_size(const struct harm::Header *header,
                                 struct PhotonArray photon,
                                 enum PhotonState *photon_state,
                                 double *step_size);

/**
 * @brief Advance photons along their trajectories by the computed step size.
 *
 * Updates photon positions and optionally accumulates path lengths.
 *
 * @param header       Pointer to simulation header.
 * @param photon       Device array of photons to propagate.
 * @param photon_state Device array of photon states.
 * @param dl           Device array to store path length increments for each photon.
 */
static __global__ void
push_photon(const struct harm::Header *header, struct PhotonArray photon, enum PhotonState *photon_state, double *dl);

/**
 * @brief Compute photon interactions with the fluid, including absorption and scattering increments.
 *
 * This kernel calculates optical depth contributions (scattering and absorption) for each photon in the simulation.
 * It updates local photon properties such as frequency in the fluid frame,  angle with the magnetic field, and photon
 * weights using bias factors. Photons outside the fluid or with zero interaction rates are skipped.
 *
 * @param header        Pointer to simulation header.
 * @param data          Device copy of simulation data (geometry, fluid, zones).
 * @param units         Pointer to unit conversion structure.
 * @param tables        Precomputed tables (hotcross, k2, f) on device memory.
 * @param photon        Device array of photons to interact.
 * @param photon_state  Device array of photon states.
 * @param interact_cond Device array of boolean flags indicating which photons interact.
 * @param step_size     Device array of photon step sizes.
 * @param bias_norm     Bias normalization factor.
 * @param fluid_n_e     Device array of local electron number densities.
 * @param theta         Device array of local electron temperatures.
 * @param nu            Device array of photon frequencies in the fluid frame.
 * @param alpha_scatti  Device array of inverse scattering opacities.
 * @param alpha_absi    Device array of inverse absorption opacities.
 * @param bi            Device array of photon bias factors.
 * @param d_tau_scatt   Device array of scattering optical depth increments.
 * @param d_tau_abs     Device array of absorption optical depth increments.
 * @param bias          Device array of updated photon biases after interaction.
 */
static __global__ void interact_photon(const struct harm::Header *header,
                                       const struct cuda_harm::Data data,
                                       const struct harm::Units *units,
                                       const struct cuda_harm::Tables tables,
                                       struct PhotonArray photon,
                                       enum PhotonState *photon_state,
                                       bool *interact_cond,
                                       double *step_size,
                                       double bias_norm,
                                       double *fluid_n_e,
                                       double *theta,
                                       double *nu,
                                       double *alpha_scatti,
                                       double *alpha_absi,
                                       double *bi,
                                       double *d_tau_scatt,
                                       double *d_tau_abs,
                                       double *bias);

/**
 * @brief Process photon scattering events including secondary photon generation.
 *
 * This kernel performs Monte Carlo scattering using the photon weight and optical depth increments. Photons may be
 * partially absorbed before scattering. Secondary photons are optionally created in `photon_2` for scattering events.
 * Geodesic propagation and fluid parameters are updated for post-scatter photon states.
 *
 * @param rng_state     Device RNG states for Monte Carlo sampling.
 * @param header        Simulation header.
 * @param data          Device copy of simulation data.
 * @param units         Pointer to unit conversion structure.
 * @param tables        Precomputed tables (hotcross, k2, f).
 * @param photon        Device array of photons to propagate.
 * @param photon_state  Device array of photon states.
 * @param interact_cond Flags indicating which photons should interact.
 * @param scatter_cond  Flags indicating which photons scatter.
 * @param photon_2      Device array for secondary photons generated by scattering.
 * @param photon_p      Device array for temporary photon storage.
 * @param fluid_params  Device array of fluid parameters at photon locations.
 * @param g_cov         Device array for metric connection coefficients.
 * @param step_size     Device array of photon step sizes.
 * @param bias_norm     Photon weight bias normalization factor.
 * @param theta         Device array of electron temperatures (theta_e).
 * @param nu            Device array of photon frequencies in the fluid frame.
 * @param alpha_scatti  Device array of inverse scattering opacities.
 * @param alpha_absi    Device array of inverse absorption opacities.
 * @param bi            Device array of photon bias factors.
 * @param d_tau_scatt   Device array of scattering optical depth increments.
 * @param d_tau_abs     Device array of absorption optical depth increments.
 * @param bias          Device array of updated photon biases after interaction.
 */
static __global__ void interact_photon_2(curandStatePhilox4_32_10_t *rng_state,
                                         const struct harm::Header *header,
                                         const struct cuda_harm::Data data,
                                         const struct harm::Units *units,
                                         const struct cuda_harm::Tables tables,
                                         struct PhotonArray photon,
                                         enum PhotonState *photon_state,
                                         bool *interact_cond,
                                         bool *scatter_cond,
                                         struct PhotonArray photon_2,
                                         struct PhotonArray photon_p,
                                         struct harm::FluidParams *fluid_params,
                                         double *g_cov,
                                         double *step_size,
                                         double bias_norm,
                                         double *theta,
                                         double *nu,
                                         double *alpha_scatti,
                                         double *alpha_absi,
                                         double *bi,
                                         double *d_tau_scatt,
                                         double *d_tau_abs,
                                         double *bias);

/**
 * @brief Scatter a photon according to local fluid properties and electron distribution.
 *
 * Updates the photon's momentum and flags after scattering.
 *
 * @param rng_state    Device RNG states.
 * @param units        Pointer to units conversion.
 * @param photon       Device array of photons to scatter.
 * @param photon_state Device array of photon states.
 * @param scatter_cond Flags indicating which photons scatter.
 * @param photon_p     Device array for temporary photon storage.
 * @param fluid_params Fluid parameters at photon locations.
 * @param g_cov        Metric connection coefficients for Lorentz transformations.
 */
static __global__ void scatter_super_photon(curandStatePhilox4_32_10_t *rng_state,
                                            const struct harm::Units *units,
                                            struct PhotonArray photon,
                                            enum PhotonState *photon_state,
                                            bool *scatter_cond,
                                            struct PhotonArray photon_p,
                                            struct harm::FluidParams *fluid_params,
                                            double *g_cov);

/**
 * @brief Increment photon step counters and check against max step number.
 *
 * @param n_step       Device array of photon step counters.
 * @param photon_state Device array of photon states, updated if stopping conditions met.
 */
static __global__ void incr_check_n_step(int *n_step, enum PhotonState *photon_state);

/**
 * @brief Record photons into the spectrum accumulator.
 *
 * Adds contributions from propagated photons to the device spectrum arrays.
 *
 * @param header       Simulation header.
 * @param photon       Device array of photons.
 * @param photon_state Device array of photon states.
 * @param n_step       Device array of photon step counts.
 * @param spectrum     Device array to accumulate photon spectra.
 */
static __global__ void record_super_photon(const struct harm::Header *header,
                                           struct PhotonArray photon,
                                           enum PhotonState *photon_state,
                                           int *n_step,
                                           struct harm::Spectrum *spectrum);

/**
 * @brief Device helper: advance a single photon along a step.
 *
 * @param header    Simulation header.
 * @param photon    Photon to propagate.
 * @param step_size Distance to propagate photon.
 */
static __device__ void push_photon(const struct harm::Header *header, struct photon::Photon *photon, double step_size);

/**
 * @brief Device helper: advance a single photon along a step.
 *
 * @param header    Simulation header.
 * @param photon    Photon to propagate.
 * @param step_size Distance to propagate photon.
 *
 * @return Energy and estimated errors.
 */
static __device__ std::tuple<double, double, double>
push_photon_step(const struct harm::Header *header, struct photon::Photon *photon, double step_size);

/**
 * @brief Compute photon weight bias factor for Monte Carlo propagation.
 *
 * @param bias_norm Bias normalization factor.
 * @param t_e       Local electron temperature (theta_e).
 * @param w         Photon weight.
 *
 * @return Photon bias factor.
 */
static __device__ double bias_func(double bias_norm, double t_e, double w);

/**
 * @brief Initialize photon momentum derivative dkdlam at given position.
 *
 * @param header Simulation header.
 * @param x      Photon position 4-vector.
 * @param k_con  Photon canonical momentum 4-vector.
 * @param d_k    Output derivative of momentum along geodesic.
 */
static __device__ void init_dkdlam(const struct harm::Header *header,
                                   const double (&x)[consts::n_dim],
                                   const double (&k_con)[consts::n_dim],
                                   double (&d_k)[consts::n_dim]);

/**
 * @brief Length of flattened connection coeffictients.
 */
constexpr int lconn_flat_len = 40;

/**
 * @brief Computes index of flattened connection coefficient from 3D index.
 *
 * @param i First dimension index.
 * @param j Second dimension index.
 * @param k Third dimension index.
 *
 * @returns Flattened index.
 */
__device__ __forceinline__ int lconn_flat_idx(int i, int j, int k) {
    if (j > k) {
        /* Enforce j <= k */
        int tmp = j;
        j = k;
        k = tmp;
    }
    /* Triangular number + offset */
    return 10 * i + j * (2 * consts::n_dim - j + 1) / 2 + (k - j);
}

/**
 * @brief Compute connection coefficients at a point for geodesic propagation.
 *
 * @param header Simulation header.
 * @param x      Position 4-vector.
 * @param lconn  Output 3D array of connection coefficients.
 */
static __device__ void
get_connection(const struct harm::Header *header, const double (&x)[consts::n_dim], double (&lconn)[lconn_flat_len]);

/**
 * @brief Sample a scattered photon momentum from the electron distribution.
 *
 * @param rng_state Device RNG states.
 * @param k         Incoming photon momentum 4-vector.
 * @param p         Electron momentum 4-vector.
 * @param kp        Output scattered photon momentum 4-vector.
 */
static __device__ void sample_scattered_photon(curandStatePhilox4_32_10_t *rng_state,
                                               const double (&k)[consts::n_dim],
                                               double (&p)[consts::n_dim],
                                               double (&kp)[consts::n_dim]);

/**
 * @brief Perform a Lorentz boost of a 4-vector from one frame to another.
 *
 * @param v  Input 4-vector in original frame.
 * @param u  Velocity 4-vector of target frame.
 * @param vp Output boosted 4-vector.
 */
static __device__ void
boost(const double (&v)[consts::n_dim], const double (&u)[consts::n_dim], double (&vp)[consts::n_dim]);

/**
 * @brief Atomic maximum for double precision numbers in device memory.
 *
 * @param addr Pointer to memory address to perform atomic max.
 * @param val  Value to compare and store if larger.
 *
 * @return Maximum value after atomic operation.
 */
static __device__ double atomic_max_double(double *addr, double val);

void alloc_memory(const struct harm::Header &header,
                  const struct harm::Data &data,
                  const struct harm::Units &units,
                  const ndarray::NDArray<double, 2> &hotcross_table,
                  const std::array<double, consts::n_e_samp + 1> &f,
                  const std::array<double, consts::n_e_samp + 1> &k2) {
    gpuErrchk(cudaMalloc((void **)&dev_header, sizeof(struct harm::Header)));
    gpuErrchk(cudaMalloc((void **)&dev_units, consts::cuda::threads_per_grid * sizeof(struct harm::Units)));
    gpuErrchk(cudaMalloc((void **)&dev_data.k_rho, sizeof(double) * data.k_rho.size()));
    gpuErrchk(cudaMalloc((void **)&dev_data.u, sizeof(double) * data.u.size()));
    gpuErrchk(cudaMalloc((void **)&dev_data.u_1, sizeof(double) * data.u_1.size()));
    gpuErrchk(cudaMalloc((void **)&dev_data.u_2, sizeof(double) * data.u_2.size()));
    gpuErrchk(cudaMalloc((void **)&dev_data.u_3, sizeof(double) * data.u_3.size()));
    gpuErrchk(cudaMalloc((void **)&dev_data.b_1, sizeof(double) * data.b_1.size()));
    gpuErrchk(cudaMalloc((void **)&dev_data.b_2, sizeof(double) * data.b_2.size()));
    gpuErrchk(cudaMalloc((void **)&dev_data.b_3, sizeof(double) * data.b_3.size()));
    gpuErrchk(cudaMalloc((void **)&dev_tables.hotcross_table, sizeof(double) * hotcross_table.size()));
    gpuErrchk(cudaMalloc((void **)&dev_tables.f, sizeof(double) * f.size()));
    gpuErrchk(cudaMalloc((void **)&dev_tables.k2, sizeof(double) * k2.size()));
    gpuErrchk(cudaMalloc((void **)&dev_spectrum, sizeof(struct harm::Spectrum) * consts::n_th_bins * consts::n_e_bins));

    gpuErrchk(cudaMemcpy(dev_header, &header, sizeof(struct harm::Header), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_units, &units, sizeof(struct harm::Units), cudaMemcpyHostToDevice));
    gpuErrchk(
        cudaMemcpy(dev_data.k_rho, data.k_rho.data(), sizeof(double) * data.k_rho.size(), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_data.u, data.u.data(), sizeof(double) * data.u.size(), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_data.u_1, data.u_1.data(), sizeof(double) * data.u_1.size(), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_data.u_2, data.u_2.data(), sizeof(double) * data.u_2.size(), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_data.u_3, data.u_3.data(), sizeof(double) * data.u_3.size(), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_data.b_1, data.b_1.data(), sizeof(double) * data.b_1.size(), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_data.b_2, data.b_2.data(), sizeof(double) * data.b_2.size(), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_data.b_3, data.b_3.data(), sizeof(double) * data.b_3.size(), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_tables.hotcross_table,
                         hotcross_table.data(),
                         sizeof(double) * hotcross_table.size(),
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_tables.f, f.data(), sizeof(double) * f.size(), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_tables.k2, k2.data(), sizeof(double) * k2.size(), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemset(dev_spectrum, 0, sizeof(harm::Spectrum) * consts::n_th_bins * consts::n_e_bins));
}

void free_memory() {
    gpuErrchk(cudaFree(dev_header));
    gpuErrchk(cudaFree(dev_units));
    gpuErrchk(cudaFree(dev_data.k_rho));
    gpuErrchk(cudaFree(dev_data.u));
    gpuErrchk(cudaFree(dev_data.u_1));
    gpuErrchk(cudaFree(dev_data.u_2));
    gpuErrchk(cudaFree(dev_data.u_3));
    gpuErrchk(cudaFree(dev_data.b_1));
    gpuErrchk(cudaFree(dev_data.b_2));
    gpuErrchk(cudaFree(dev_data.b_3));
    gpuErrchk(cudaFree(dev_tables.hotcross_table));
    gpuErrchk(cudaFree(dev_tables.f));
    gpuErrchk(cudaFree(dev_tables.k2));
    gpuErrchk(cudaFree(dev_spectrum));
}

void track_super_photons(double bias_norm,
                         double max_tau_scatt,
                         utils::ConcurrentQueue<photon::InitPhoton> &photon_queue,
                         std::binary_semaphore &stop_sem,
                         harm::Spectrum (&spectrum)[consts::n_th_bins][consts::n_e_bins],
                         uint64_t &n_super_photon_recorded,
                         uint64_t &n_super_photon_scatt) {
    const int grid_dim = consts::cuda::grid_dim;
    const int block_dim = consts::cuda::block_dim;

    constexpr unsigned int n_streams = 2;

    struct PhotonArray photon_new[n_streams];
    enum PhotonState *photon_state[n_streams];

    curandStatePhilox4_32_10_t *dev_rng_state[n_streams];

    struct PhotonArray dev_photon[n_streams];
    enum PhotonState *dev_photon_state[n_streams];
    struct PhotonArray dev_photon_new[n_streams];
    struct PhotonArray dev_photon_2[n_streams];

    int *dev_n_step[n_streams];

    double *dev_fluid_n_e[n_streams];

    double *dev_theta[n_streams];
    double *dev_nu[n_streams];
    double *dev_alpha_scatti[n_streams];
    double *dev_alpha_absi[n_streams];
    double *dev_bi[n_streams];

    double *dev_step_size[n_streams];

    bool *dev_interact_cond[n_streams];
    bool *scatter_cond[n_streams];

    bool *dev_scatter_cond[n_streams];
    double *dev_d_tau_scatt[n_streams];
    double *dev_d_tau_abs[n_streams];
    double *dev_bias[n_streams];

    struct PhotonArray photon_p[n_streams];

    struct PhotonArray dev_photon_p[n_streams];
    struct harm::FluidParams *dev_fluid_params[n_streams];
    double *dev_g_cov[n_streams];

    gpuErrchk(cudaMemcpyToSymbol(dev_max_tau_scatt, &max_tau_scatt, sizeof(double)));

    for (int i = 0; i < n_streams; ++i) {
        gpuErrchk(cudaMalloc((void **)&dev_rng_state[i], n_photons * sizeof(curandStatePhilox4_32_10_t)));

        gpuErrchk(cudaMallocHost((void **)&photon_state[i], n_photons * sizeof(enum PhotonState)));

        for (int j = 0; j < consts::n_dim; ++j) {
            photon_new[i].x[j] = new double[n_photons];
            photon_new[i].k[j] = new double[n_photons];
        }
        photon_new[i].w = new double[n_photons];
        photon_new[i].e = new double[n_photons];
        photon_new[i].l = new double[n_photons];
        photon_new[i].n_e_0 = new double[n_photons];
        photon_new[i].theta_e_0 = new double[n_photons];
        photon_new[i].b_0 = new double[n_photons];
        photon_new[i].e_0 = new double[n_photons];
        photon_new[i].n_scatt = new int[n_photons];

        for (int j = 0; j < n_photons; ++j) {
            photon_state[i][j] = PhotonState::Empty;
        }

        gpuErrchk(cudaMallocHost((void **)&scatter_cond[i], n_photons * sizeof(bool)));

        for (int j = 0; j < consts::n_dim; ++j) {
            gpuErrchk(cudaMallocHost((void **)&photon_p[i].x[j], n_photons * sizeof(double)));
            gpuErrchk(cudaMallocHost((void **)&photon_p[i].k[j], n_photons * sizeof(double)));
        }
        gpuErrchk(cudaMallocHost((void **)&photon_p[i].w, n_photons * sizeof(double)));
        gpuErrchk(cudaMallocHost((void **)&photon_p[i].e, n_photons * sizeof(double)));
        gpuErrchk(cudaMallocHost((void **)&photon_p[i].l, n_photons * sizeof(double)));
        gpuErrchk(cudaMallocHost((void **)&photon_p[i].n_e_0, n_photons * sizeof(double)));
        gpuErrchk(cudaMallocHost((void **)&photon_p[i].b_0, n_photons * sizeof(double)));
        gpuErrchk(cudaMallocHost((void **)&photon_p[i].theta_e_0, n_photons * sizeof(double)));
        gpuErrchk(cudaMallocHost((void **)&photon_p[i].e_0, n_photons * sizeof(double)));
        gpuErrchk(cudaMallocHost((void **)&photon_p[i].n_scatt, n_photons * sizeof(int)));

        /* TODO: optimize memory usage by allocating only the parts that are needed */
        alloc_photon_array(dev_photon[i], n_photons);
        gpuErrchk(cudaMalloc((void **)&dev_photon_state[i], n_photons * sizeof(enum PhotonState)));
        alloc_photon_array(dev_photon_new[i], n_photons);
        alloc_photon_array(dev_photon_2[i], n_photons);
        gpuErrchk(cudaMalloc((void **)&dev_n_step[i], n_photons * sizeof(int)));

        gpuErrchk(cudaMalloc((void **)&dev_fluid_n_e[i], n_photons * sizeof(double)));

        gpuErrchk(cudaMalloc((void **)&dev_theta[i], n_photons * sizeof(double)));
        gpuErrchk(cudaMalloc((void **)&dev_nu[i], n_photons * sizeof(double)));
        gpuErrchk(cudaMalloc((void **)&dev_alpha_scatti[i], n_photons * sizeof(double)));
        gpuErrchk(cudaMalloc((void **)&dev_alpha_absi[i], n_photons * sizeof(double)));
        gpuErrchk(cudaMalloc((void **)&dev_bi[i], n_photons * sizeof(double)));

        gpuErrchk(cudaMalloc((void **)&dev_step_size[i], n_photons * sizeof(double)));

        gpuErrchk(cudaMalloc((void **)&dev_interact_cond[i], n_photons * sizeof(bool)));
        gpuErrchk(cudaMalloc((void **)&dev_scatter_cond[i], n_photons * sizeof(bool)));
        gpuErrchk(cudaMalloc((void **)&dev_d_tau_scatt[i], n_photons * sizeof(double)));
        gpuErrchk(cudaMalloc((void **)&dev_d_tau_abs[i], n_photons * sizeof(double)));
        gpuErrchk(cudaMalloc((void **)&dev_bias[i], n_photons * sizeof(double)));

        alloc_photon_array(dev_photon_p[i], n_photons);
        gpuErrchk(cudaMalloc((void **)&dev_fluid_params[i], n_photons * sizeof(struct harm::FluidParams)));
        gpuErrchk(cudaMalloc((void **)&dev_g_cov[i], n_photons * consts::n_dim * consts::n_dim * sizeof(double)));

        gpuErrchk(cudaMemset(dev_photon_state[i], 0, n_photons * sizeof(enum PhotonState)));
    }

    int n_iter = 0;
    bool queue_empty = false;
    bool all_done = false;

    cudaStream_t streams[n_streams];
    cudaEvent_t scattered_photons_cpy_dtoh[n_streams];
    cudaEvent_t scattered_photons_enq[n_streams];
    int stream_idx = 0;

    for (auto &stream : streams) {
        cudaStreamCreate(&stream);
    }
    for (auto &event : scattered_photons_cpy_dtoh) {
        cudaEventCreate(&event);
    }
    for (auto &event : scattered_photons_enq) {
        cudaEventCreate(&event);
    }

    for (int i = 0; i < n_streams; ++i) {
        init_rng<<<grid_dim, block_dim, 0, streams[i]>>>(dev_rng_state[i]);
    }

    gpuErrchk(cudaDeviceSynchronize());

    std::queue<photon::InitPhoton> buffer;

    while (true) {
        if (stop_sem.try_acquire()) {
            queue_empty = true;
        }

        /* feed photons into array */
        all_done = true;
        if (n_iter % 7 == 0) {
            photon_queue.dequeue_n(buffer, consts::cuda::n_photons - buffer.size());

            for (int i = 0; i < n_photons; ++i) {
                if (photon_state[stream_idx][i] == PhotonState::Empty && !buffer.empty()) {
                    photon::InitPhoton p = buffer.front();
                    buffer.pop();
                    for (int j = 0; j < consts::n_dim; ++j) {
                        photon_new[stream_idx].x[j][i] = p.x[j];
                        photon_new[stream_idx].k[j][i] = p.k[j];
                    }
                    photon_new[stream_idx].w[i] = p.w;
                    photon_new[stream_idx].e[i] = p.e;
                    photon_new[stream_idx].l[i] = p.l;
                    photon_new[stream_idx].n_e_0[i] = p.n_e_0;
                    photon_new[stream_idx].b_0[i] = p.b_0;
                    photon_new[stream_idx].theta_e_0[i] = p.theta_e_0;
                    photon_new[stream_idx].e_0[i] = p.e_0;
                    photon_new[stream_idx].n_scatt[i] = p.n_scatt;
                    photon_state[stream_idx][i] = PhotonState::New;
                }
                if (photon_state[stream_idx][i] != PhotonState::Empty) {
                    all_done = false;
                }
            }

            if (queue_empty && all_done) {
                break;
            }

            /* load and validate new photons */
            for (int i = 0; i < consts::n_dim; ++i) {
                gpuErrchk(cudaMemcpyAsync(dev_photon_new[stream_idx].x[i],
                                          photon_new[stream_idx].x[i],
                                          n_photons * sizeof(double),
                                          cudaMemcpyHostToDevice,
                                          streams[stream_idx]));
                gpuErrchk(cudaMemcpyAsync(dev_photon_new[stream_idx].k[i],
                                          photon_new[stream_idx].k[i],
                                          n_photons * sizeof(double),
                                          cudaMemcpyHostToDevice,
                                          streams[stream_idx]));
            }
            gpuErrchk(cudaMemcpyAsync(dev_photon_new[stream_idx].w,
                                      photon_new[stream_idx].w,
                                      n_photons * sizeof(double),
                                      cudaMemcpyHostToDevice,
                                      streams[stream_idx]));
            gpuErrchk(cudaMemcpyAsync(dev_photon_new[stream_idx].e,
                                      photon_new[stream_idx].e,
                                      n_photons * sizeof(double),
                                      cudaMemcpyHostToDevice,
                                      streams[stream_idx]));
            gpuErrchk(cudaMemcpyAsync(dev_photon_new[stream_idx].l,
                                      photon_new[stream_idx].l,
                                      n_photons * sizeof(double),
                                      cudaMemcpyHostToDevice,
                                      streams[stream_idx]));
            gpuErrchk(cudaMemcpyAsync(dev_photon_new[stream_idx].n_e_0,
                                      photon_new[stream_idx].n_e_0,
                                      n_photons * sizeof(double),
                                      cudaMemcpyHostToDevice,
                                      streams[stream_idx]));
            gpuErrchk(cudaMemcpyAsync(dev_photon_new[stream_idx].b_0,
                                      photon_new[stream_idx].b_0,
                                      n_photons * sizeof(double),
                                      cudaMemcpyHostToDevice,
                                      streams[stream_idx]));
            gpuErrchk(cudaMemcpyAsync(dev_photon_new[stream_idx].theta_e_0,
                                      photon_new[stream_idx].theta_e_0,
                                      n_photons * sizeof(double),
                                      cudaMemcpyHostToDevice,
                                      streams[stream_idx]));
            gpuErrchk(cudaMemcpyAsync(dev_photon_new[stream_idx].e_0,
                                      photon_new[stream_idx].e_0,
                                      n_photons * sizeof(double),
                                      cudaMemcpyHostToDevice,
                                      streams[stream_idx]));
            gpuErrchk(cudaMemcpyAsync(dev_photon_new[stream_idx].n_scatt,
                                      photon_new[stream_idx].n_scatt,
                                      n_photons * sizeof(int),
                                      cudaMemcpyHostToDevice,
                                      streams[stream_idx]));

            gpuErrchk(cudaMemcpyAsync(dev_photon_state[stream_idx],
                                      &photon_state[stream_idx][0],
                                      n_photons * sizeof(enum PhotonState),
                                      cudaMemcpyHostToDevice,
                                      streams[stream_idx]));

            load_validate_photon<<<grid_dim, block_dim, 0, streams[stream_idx]>>>(
                dev_photon[stream_idx], dev_photon_new[stream_idx], dev_photon_state[stream_idx]);

            setup_variables<<<grid_dim, block_dim, 0, streams[stream_idx]>>>(dev_header,
                                                                             dev_data,
                                                                             dev_units,
                                                                             dev_tables,
                                                                             bias_norm,
                                                                             dev_photon[stream_idx],
                                                                             dev_photon_state[stream_idx],
                                                                             dev_n_step[stream_idx],
                                                                             dev_fluid_n_e[stream_idx],
                                                                             dev_theta[stream_idx],
                                                                             dev_nu[stream_idx],
                                                                             dev_alpha_scatti[stream_idx],
                                                                             dev_alpha_absi[stream_idx],
                                                                             dev_bi[stream_idx]);
        }
        ++n_iter;

        stop_criterion<<<grid_dim, block_dim, 0, streams[stream_idx]>>>(
            dev_rng_state[stream_idx], dev_header, dev_photon[stream_idx], dev_photon_state[stream_idx]);

        for (int i = 0; i < consts::n_dim; ++i) {
            gpuErrchk(cudaMemcpyAsync(dev_photon_2[stream_idx].x[i],
                                      dev_photon[stream_idx].x[i],
                                      n_photons * sizeof(double),
                                      cudaMemcpyDeviceToDevice,
                                      streams[stream_idx]));
            gpuErrchk(cudaMemcpyAsync(dev_photon_2[stream_idx].k[i],
                                      dev_photon[stream_idx].k[i],
                                      n_photons * sizeof(double),
                                      cudaMemcpyDeviceToDevice,
                                      streams[stream_idx]));
            gpuErrchk(cudaMemcpyAsync(dev_photon_2[stream_idx].dkdlam[i],
                                      dev_photon[stream_idx].dkdlam[i],
                                      n_photons * sizeof(double),
                                      cudaMemcpyDeviceToDevice,
                                      streams[stream_idx]));
        }
        gpuErrchk(cudaMemcpyAsync(dev_photon_2[stream_idx].e_0_s,
                                  dev_photon[stream_idx].e_0_s,
                                  n_photons * sizeof(double),
                                  cudaMemcpyDeviceToDevice,
                                  streams[stream_idx]));

        step_size<<<grid_dim, block_dim, 0, streams[stream_idx]>>>(
            dev_header, dev_photon[stream_idx], dev_photon_state[stream_idx], dev_step_size[stream_idx]);

        push_photon<<<grid_dim, block_dim, 0, streams[stream_idx]>>>(
            dev_header, dev_photon[stream_idx], dev_photon_state[stream_idx], dev_step_size[stream_idx]);

        /* check stop criterion */
        stop_criterion<<<grid_dim, block_dim, 0, streams[stream_idx]>>>(
            dev_rng_state[stream_idx], dev_header, dev_photon[stream_idx], dev_photon_state[stream_idx]);

        /* allow photon to interact with matter */
        interact_photon<<<grid_dim, block_dim, 0, streams[stream_idx]>>>(dev_header,
                                                                         dev_data,
                                                                         dev_units,
                                                                         dev_tables,
                                                                         dev_photon[stream_idx],
                                                                         dev_photon_state[stream_idx],
                                                                         dev_interact_cond[stream_idx],
                                                                         dev_step_size[stream_idx],
                                                                         bias_norm,
                                                                         dev_fluid_n_e[stream_idx],
                                                                         dev_theta[stream_idx],
                                                                         dev_nu[stream_idx],
                                                                         dev_alpha_scatti[stream_idx],
                                                                         dev_alpha_absi[stream_idx],
                                                                         dev_bi[stream_idx],
                                                                         dev_d_tau_scatt[stream_idx],
                                                                         dev_d_tau_abs[stream_idx],
                                                                         dev_bias[stream_idx]);

        interact_photon_2<<<grid_dim, block_dim, 0, streams[stream_idx]>>>(dev_rng_state[stream_idx],
                                                                           dev_header,
                                                                           dev_data,
                                                                           dev_units,
                                                                           dev_tables,
                                                                           dev_photon[stream_idx],
                                                                           dev_photon_state[stream_idx],
                                                                           dev_interact_cond[stream_idx],
                                                                           dev_scatter_cond[stream_idx],
                                                                           dev_photon_2[stream_idx],
                                                                           dev_photon_p[stream_idx],
                                                                           dev_fluid_params[stream_idx],
                                                                           dev_g_cov[stream_idx],
                                                                           dev_step_size[stream_idx],
                                                                           bias_norm,
                                                                           dev_theta[stream_idx],
                                                                           dev_nu[stream_idx],
                                                                           dev_alpha_scatti[stream_idx],
                                                                           dev_alpha_absi[stream_idx],
                                                                           dev_bi[stream_idx],
                                                                           dev_d_tau_scatt[stream_idx],
                                                                           dev_d_tau_abs[stream_idx],
                                                                           dev_bias[stream_idx]);

        /* scatter */
        scatter_super_photon<<<grid_dim, block_dim, 0, streams[stream_idx]>>>(dev_rng_state[stream_idx],
                                                                              dev_units,
                                                                              dev_photon[stream_idx],
                                                                              dev_photon_state[stream_idx],
                                                                              dev_scatter_cond[stream_idx],
                                                                              dev_photon_p[stream_idx],
                                                                              dev_fluid_params[stream_idx],
                                                                              dev_g_cov[stream_idx]);

        cudaEventSynchronize(scattered_photons_enq[stream_idx]);

        for (int i = 0; i < consts::n_dim; ++i) {
            gpuErrchk(cudaMemcpyAsync(photon_p[stream_idx].x[i],
                                      dev_photon_p[stream_idx].x[i],
                                      n_photons * sizeof(double),
                                      cudaMemcpyDeviceToHost,
                                      streams[stream_idx]));
            gpuErrchk(cudaMemcpyAsync(photon_p[stream_idx].k[i],
                                      dev_photon_p[stream_idx].k[i],
                                      n_photons * sizeof(double),
                                      cudaMemcpyDeviceToHost,
                                      streams[stream_idx]));
        }
        gpuErrchk(cudaMemcpyAsync(photon_p[stream_idx].w,
                                  dev_photon_p[stream_idx].w,
                                  n_photons * sizeof(double),
                                  cudaMemcpyDeviceToHost,
                                  streams[stream_idx]));
        gpuErrchk(cudaMemcpyAsync(photon_p[stream_idx].e,
                                  dev_photon_p[stream_idx].e,
                                  n_photons * sizeof(double),
                                  cudaMemcpyDeviceToHost,
                                  streams[stream_idx]));
        gpuErrchk(cudaMemcpyAsync(photon_p[stream_idx].l,
                                  dev_photon_p[stream_idx].l,
                                  n_photons * sizeof(double),
                                  cudaMemcpyDeviceToHost,
                                  streams[stream_idx]));
        gpuErrchk(cudaMemcpyAsync(photon_p[stream_idx].n_e_0,
                                  dev_photon_p[stream_idx].n_e_0,
                                  n_photons * sizeof(double),
                                  cudaMemcpyDeviceToHost,
                                  streams[stream_idx]));
        gpuErrchk(cudaMemcpyAsync(photon_p[stream_idx].b_0,
                                  dev_photon_p[stream_idx].b_0,
                                  n_photons * sizeof(double),
                                  cudaMemcpyDeviceToHost,
                                  streams[stream_idx]));
        gpuErrchk(cudaMemcpyAsync(photon_p[stream_idx].theta_e_0,
                                  dev_photon_p[stream_idx].theta_e_0,
                                  n_photons * sizeof(double),
                                  cudaMemcpyDeviceToHost,
                                  streams[stream_idx]));
        gpuErrchk(cudaMemcpyAsync(photon_p[stream_idx].e_0,
                                  dev_photon_p[stream_idx].e_0,
                                  n_photons * sizeof(double),
                                  cudaMemcpyDeviceToHost,
                                  streams[stream_idx]));
        gpuErrchk(cudaMemcpyAsync(photon_p[stream_idx].n_scatt,
                                  dev_photon_p[stream_idx].n_scatt,
                                  n_photons * sizeof(int),
                                  cudaMemcpyDeviceToHost,
                                  streams[stream_idx]));

        gpuErrchk(cudaMemcpyAsync(scatter_cond[stream_idx],
                                  dev_scatter_cond[stream_idx],
                                  n_photons * sizeof(bool),
                                  cudaMemcpyDeviceToHost,
                                  streams[stream_idx]));

        gpuErrchk(cudaEventRecord(scattered_photons_cpy_dtoh[stream_idx], streams[stream_idx]));

        /* increment and check step num */
        incr_check_n_step<<<grid_dim, block_dim, 0, streams[stream_idx]>>>(dev_n_step[stream_idx],
                                                                           dev_photon_state[stream_idx]);

        if (n_iter % 7 == 0) {
            /* record photons */
            record_super_photon<<<grid_dim, block_dim, 0, streams[stream_idx]>>>(
                dev_header, dev_photon[stream_idx], dev_photon_state[stream_idx], dev_n_step[stream_idx], dev_spectrum);

            /* copy photon state to host */
            gpuErrchk(cudaMemcpyAsync(&photon_state[stream_idx][0],
                                      dev_photon_state[stream_idx],
                                      n_photons * sizeof(enum PhotonState),
                                      cudaMemcpyDeviceToHost,
                                      streams[stream_idx]));
        }

        unsigned int prev_stream_idx = (stream_idx + n_streams - 1) % n_streams;
        cudaEventSynchronize(scattered_photons_cpy_dtoh[prev_stream_idx]);

        for (int i = 0; i < n_photons; ++i) {
            if (scatter_cond[prev_stream_idx][i]) {
                photon::InitPhoton p;

                for (int j = 0; j < consts::n_dim; ++j) {
                    p.x[j] = photon_p[prev_stream_idx].x[j][i];
                    p.k[j] = photon_p[prev_stream_idx].k[j][i];
                }
                p.w = photon_p[prev_stream_idx].w[i];
                p.e = photon_p[prev_stream_idx].e[i];
                p.l = photon_p[prev_stream_idx].l[i];
                p.n_e_0 = photon_p[prev_stream_idx].n_e_0[i];
                p.b_0 = photon_p[prev_stream_idx].b_0[i];
                p.theta_e_0 = photon_p[prev_stream_idx].theta_e_0[i];
                p.e_0 = photon_p[prev_stream_idx].e_0[i];
                p.n_scatt = photon_p[prev_stream_idx].n_scatt[i];

                photon_queue.force_enqueue(p);
            }
        }

        gpuErrchk(cudaEventRecord(scattered_photons_enq[stream_idx], streams[stream_idx]));

        ++stream_idx;
        stream_idx %= n_streams;
    }

    for (auto &stream : streams) {
        cudaStreamDestroy(stream);
    }
    for (auto &event : scattered_photons_cpy_dtoh) {
        cudaEventDestroy(event);
    }
    for (auto &event : scattered_photons_enq) {
        cudaEventDestroy(event);
    }

    gpuErrchk(cudaMemcpy(
        spectrum, dev_spectrum, sizeof(harm::Spectrum) * consts::n_th_bins * consts::n_e_bins, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpyFromSymbol(&n_super_photon_recorded, dev_n_super_photon_recorded, sizeof(int)));
    gpuErrchk(cudaMemcpyFromSymbol(&n_super_photon_scatt, dev_n_super_photon_scatt, sizeof(int)));

    for (int i = 0; i < n_streams; ++i) {
        gpuErrchk(cudaFree(dev_rng_state[i]));

        gpuErrchk(cudaFreeHost(photon_state[i]));

        for (int j = 0; j < consts::n_dim; ++j) {
            delete[] photon_new[i].x[j];
            delete[] photon_new[i].k[j];
        }
        delete[] photon_new[i].w;
        delete[] photon_new[i].e;
        delete[] photon_new[i].l;
        delete[] photon_new[i].n_e_0;
        delete[] photon_new[i].b_0;
        delete[] photon_new[i].theta_e_0;

        free_photon_array(dev_photon[i]);
        gpuErrchk(cudaFree(dev_photon_state[i]));
        free_photon_array(dev_photon_new[i]);
        free_photon_array(dev_photon_2[i]);
        gpuErrchk(cudaFree(dev_n_step[i]));

        gpuErrchk(cudaFree(dev_fluid_n_e[i]));

        gpuErrchk(cudaFree(dev_theta[i]));
        gpuErrchk(cudaFree(dev_nu[i]));
        gpuErrchk(cudaFree(dev_alpha_scatti[i]));
        gpuErrchk(cudaFree(dev_alpha_absi[i]));
        gpuErrchk(cudaFree(dev_bi[i]));

        gpuErrchk(cudaFree(dev_step_size[i]));

        gpuErrchk(cudaFree(dev_interact_cond[i]));
        gpuErrchk(cudaFreeHost(scatter_cond[i]));
        gpuErrchk(cudaFree(dev_scatter_cond[i]));
        gpuErrchk(cudaFree(dev_d_tau_scatt[i]));
        gpuErrchk(cudaFree(dev_d_tau_abs[i]));
        gpuErrchk(cudaFree(dev_bias[i]));

        for (int j = 0; j < consts::n_dim; ++j) {
            gpuErrchk(cudaFreeHost(photon_p[i].x[j]));
            gpuErrchk(cudaFreeHost(photon_p[i].k[j]));
        }
        gpuErrchk(cudaFreeHost(photon_p[i].w));
        gpuErrchk(cudaFreeHost(photon_p[i].e));
        gpuErrchk(cudaFreeHost(photon_p[i].l));
        gpuErrchk(cudaFreeHost(photon_p[i].n_e_0));
        gpuErrchk(cudaFreeHost(photon_p[i].b_0));
        gpuErrchk(cudaFreeHost(photon_p[i].theta_e_0));
        gpuErrchk(cudaFreeHost(photon_p[i].e_0));
        gpuErrchk(cudaFreeHost(photon_p[i].n_scatt));

        free_photon_array(dev_photon_p[i]);
        gpuErrchk(cudaFree(dev_fluid_params[i]));
        gpuErrchk(cudaFree(dev_g_cov[i]));
    }
}

static __global__ void init_rng(curandStatePhilox4_32_10_t *rng_state) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    curand_init(123, tid, 0, &rng_state[tid]);
}

static __global__ void
load_validate_photon(struct PhotonArray photon, struct PhotonArray photon_new, enum PhotonState *photon_state) {
    for (int tid = threadIdx.x + blockIdx.x * blockDim.x; tid < n_photons; tid += blockDim.x * gridDim.x) {
        if (photon_state[tid] != PhotonState::New) {
            continue;
        }

#pragma unroll
        for (int i = 0; i < consts::n_dim; ++i) {
            photon.x[i][tid] = photon_new.x[i][tid];
            photon.k[i][tid] = photon_new.k[i][tid];
        }
        photon.w[tid] = photon_new.w[tid];
        photon.e[tid] = photon_new.e[tid];
        photon.e_0[tid] = photon_new.e_0[tid];
        photon.e_0_s[tid] = photon_new.e[tid];
        photon.l[tid] = photon_new.l[tid];
        photon.tau_scatt[tid] = 0.0;
        photon.tau_abs[tid] = 0.0;
        photon.x1i[tid] = photon_new.x[1][tid];
        photon.x2i[tid] = photon_new.x[2][tid];
        photon.n_e_0[tid] = photon_new.n_e_0[tid];
        photon.b_0[tid] = photon_new.b_0[tid];
        photon.theta_e_0[tid] = photon_new.theta_e_0[tid];
        photon.n_scatt[tid] = photon_new.n_scatt[tid];

        if (isnan(photon.x[0][tid]) || isnan(photon.x[1][tid]) || isnan(photon.x[2][tid]) || isnan(photon.x[3][tid]) ||
            isnan(photon.k[0][tid]) || isnan(photon.k[1][tid]) || isnan(photon.k[2][tid]) || isnan(photon.k[3][tid]) ||
            photon.w[tid] == 0.0) {
            photon_state[tid] = PhotonState::Empty;
        }
    }
}

static __global__ void setup_variables(const struct harm::Header *header,
                                       const struct cuda_harm::Data data,
                                       const struct harm::Units *units,
                                       const struct cuda_harm::Tables tables,
                                       double bias_norm,
                                       struct PhotonArray photon,
                                       enum PhotonState *photon_state,
                                       int *n_step,
                                       double *fluid_n_e,
                                       double *theta,
                                       double *nu,
                                       double *alpha_scatti,
                                       double *alpha_absi,
                                       double *bi) {
    __shared__ double g_cov[consts::cuda::block_dim][consts::n_dim][consts::n_dim];
    for (int tid = threadIdx.x + blockIdx.x * blockDim.x; tid < n_photons; tid += blockDim.x * gridDim.x) {
        if (photon_state[tid] != PhotonState::New) {
            continue;
        }

        const double photon_x[consts::n_dim] = {photon.x[0][tid], photon.x[1][tid], photon.x[2][tid], photon.x[3][tid]};
        const double photon_k[consts::n_dim] = {photon.k[0][tid], photon.k[1][tid], photon.k[2][tid], photon.k[3][tid]};

        cuda_harm::gcov_func(header, photon_x, g_cov[threadIdx.x]);

        harm::FluidParams fluid_params = cuda_harm::get_fluid_params(header,
                                                                     units,
                                                                     data.k_rho,
                                                                     data.u,
                                                                     data.u_1,
                                                                     data.u_2,
                                                                     data.u_3,
                                                                     data.b_1,
                                                                     data.b_2,
                                                                     data.b_3,
                                                                     photon_x,
                                                                     g_cov[threadIdx.x]);

        fluid_n_e[tid] = fluid_params.n_e;

        theta[tid] = cuda_radiation::bk_angle(
            photon_x, photon_k, fluid_params.u_cov, fluid_params.b_cov, fluid_params.b, units->b_unit);
        nu[tid] = cuda_radiation::fluid_nu(photon_x, photon_k, fluid_params.u_cov);
        alpha_scatti[tid] =
            cuda_radiation::alpha_inv_scatt(nu[tid], fluid_params.theta_e, fluid_params.n_e, tables.hotcross_table);
        alpha_absi[tid] = cuda_radiation::alpha_inv_abs(
            nu[tid], fluid_params.theta_e, fluid_params.n_e, fluid_params.b, theta[tid], tables.k2);
        bi[tid] = bias_func(bias_norm, fluid_params.theta_e, photon.w[tid]);

        double photon_dkdlam[consts::n_dim];
        init_dkdlam(header, photon_x, photon_k, photon_dkdlam);

#pragma unroll
        for (int i = 0; i < consts::n_dim; ++i) {
            photon.dkdlam[i][tid] = photon_dkdlam[i];
        }

        n_step[tid] = 0;
        photon_state[tid] = PhotonState::Initialized;
    }
}

static __global__ void stop_criterion(curandStatePhilox4_32_10_t *rng_state,
                                      const struct harm::Header *header,
                                      struct PhotonArray photon,
                                      enum PhotonState *photon_state) {
    double rh_ = 1.0 + sqrt(1.0 - header->a * header->a);
    double x1_min_ = log(rh_);
    double x1_max = log(consts::r_max);
    for (int tid = threadIdx.x + blockIdx.x * blockDim.x; tid < n_photons; tid += blockDim.x * gridDim.x) {
        if (photon_state[tid] != PhotonState::Initialized) {
            continue;
        }

        /* TODO: reduce branching */
        if (photon.x[1][tid] < x1_min_) {
            /* stop at event horizon */
            photon_state[tid] = PhotonState::Tracked;
            continue;
        }

        if (photon.x[1][tid] > x1_max) {
            /* stop at large distance */
            if (photon.w[tid] < consts::weight_min) {
                if (curand_uniform(&rng_state[tid]) <= 1.0 / consts::roulette) {
                    photon.w[tid] *= consts::roulette;
                } else {
                    photon.w[tid] = 0.0;
                }
            }
            photon_state[tid] = PhotonState::Tracked;
            continue;
        }

        if (photon.w[tid] < consts::weight_min) {
            if (curand_uniform(&rng_state[tid]) <= 1.0 / consts::roulette) {
                photon.w[tid] *= consts::roulette;
            } else {
                photon.w[tid] = 0.0;
                photon_state[tid] = PhotonState::Tracked;
                continue;
            }
        }
    }
}

static __global__ void step_size(const struct harm::Header *header,
                                 struct PhotonArray photon,
                                 enum PhotonState *photon_state,
                                 double *step_size) {
    for (int tid = threadIdx.x + blockIdx.x * blockDim.x; tid < n_photons; tid += blockDim.x * gridDim.x) {
        if (photon_state[tid] != PhotonState::Initialized) {
            continue;
        }

        double dl_x_1 = consts::step_eps * photon.x[1][tid] / (fabs(photon.k[1][tid]) + consts::eps);
        double dl_x_2 = consts::step_eps * fmin(photon.x[2][tid], header->x_stop[2] - photon.x[2][tid]) /
                        (fabs(photon.k[2][tid]) + consts::eps);
        double dl_x_3 = consts::step_eps / (fabs(photon.k[3][tid]) + consts::eps);

        double i_dl_x_1 = 1.0 / (fabs(dl_x_1) + consts::eps);
        double i_dl_x_2 = 1.0 / (fabs(dl_x_2) + consts::eps);
        double i_dl_x_3 = 1.0 / (fabs(dl_x_3) + consts::eps);

        step_size[tid] = 1.0 / (i_dl_x_1 + i_dl_x_2 + i_dl_x_3);
    }
}

static __global__ void push_photon(const struct harm::Header *header,
                                   struct PhotonArray photon,
                                   enum PhotonState *photon_state,
                                   double *step_size) {
    for (int tid = threadIdx.x + blockIdx.x * blockDim.x; tid < n_photons; tid += blockDim.x * gridDim.x) {
        if (photon_state[tid] != PhotonState::Initialized) {
            continue;
        }

        struct photon::Photon p = {
            .x = {photon.x[0][tid], photon.x[1][tid], photon.x[2][tid], photon.x[3][tid]},
            .k = {photon.k[0][tid], photon.k[1][tid], photon.k[2][tid], photon.k[3][tid]},
            .dkdlam = {photon.dkdlam[0][tid], photon.dkdlam[1][tid], photon.dkdlam[2][tid], photon.dkdlam[3][tid]},
            .e_0_s = photon.e_0_s[tid],
        };

        push_photon(header, &p, step_size[tid]);

#pragma unroll
        for (int i = 0; i < consts::n_dim; ++i) {
            photon.x[i][tid] = p.x[i];
            photon.k[i][tid] = p.k[i];
            photon.dkdlam[i][tid] = p.dkdlam[i];
        }
        photon.e_0_s[tid] = p.e_0_s;
    }
}

static __global__ void interact_photon(const struct harm::Header *header,
                                       const struct cuda_harm::Data data,
                                       const struct harm::Units *units,
                                       const struct cuda_harm::Tables tables,
                                       struct PhotonArray photon,
                                       enum PhotonState *photon_state,
                                       bool *interact_cond,
                                       double *step_size,
                                       double bias_norm,
                                       double *fluid_n_e,
                                       double *theta,
                                       double *nu,
                                       double *alpha_scatti,
                                       double *alpha_absi,
                                       double *bi,
                                       double *d_tau_scatt,
                                       double *d_tau_abs,
                                       double *bias) {
    const double hbar = consts::hpl / (2.0 * CUDART_PI);
    const double d_tau_k = 2.0 * CUDART_PI * units->l_unit / (consts::me * consts::cl * consts::cl / hbar);

    __shared__ double g_cov[consts::cuda::block_dim][consts::n_dim][consts::n_dim];

    for (int tid = threadIdx.x + blockIdx.x * blockDim.x; tid < n_photons; tid += blockDim.x * gridDim.x) {
        const double photon_x[consts::n_dim] = {photon.x[0][tid], photon.x[1][tid], photon.x[2][tid], photon.x[3][tid]};
        const double photon_k[consts::n_dim] = {photon.k[0][tid], photon.k[1][tid], photon.k[2][tid], photon.k[3][tid]};

        if (photon_state[tid] != PhotonState::Initialized) {
            continue;
        }

        interact_cond[tid] = (alpha_absi[tid] > 0.0 || alpha_scatti[tid] > 0.0 || fluid_n_e[tid] > 0.0);

        if (!interact_cond[tid]) {
            continue;
        }

        cuda_harm::gcov_func(header, photon_x, g_cov[threadIdx.x]);

        harm::FluidParams fluid_params = cuda_harm::get_fluid_params(header,
                                                                     units,
                                                                     data.k_rho,
                                                                     data.u,
                                                                     data.u_1,
                                                                     data.u_2,
                                                                     data.u_3,
                                                                     data.b_1,
                                                                     data.b_2,
                                                                     data.b_3,
                                                                     photon_x,
                                                                     g_cov[threadIdx.x]);
        bool bound_flag = fluid_params.n_e == 0.0;

        if (!bound_flag) {
            theta[tid] = cuda_radiation::bk_angle(
                photon_x, photon_k, fluid_params.u_cov, fluid_params.b_cov, fluid_params.b, units->b_unit);
            nu[tid] = cuda_radiation::fluid_nu(photon_x, photon_k, fluid_params.u_cov);
        }

        if (bound_flag || (nu[tid] < 0.0)) {
            d_tau_scatt[tid] = 0.5 * alpha_scatti[tid] * d_tau_k * step_size[tid];
            d_tau_abs[tid] = 0.5 * alpha_absi[tid] * d_tau_k * step_size[tid];
            alpha_scatti[tid] = 0.0;
            alpha_absi[tid] = 0.0;
            bias[tid] = 0.0;
            bi[tid] = 0.0;
        } else {
            double alpha_scattf =
                cuda_radiation::alpha_inv_scatt(nu[tid], fluid_params.theta_e, fluid_params.n_e, tables.hotcross_table);
            d_tau_scatt[tid] = 0.5 * (alpha_scatti[tid] + alpha_scattf) * d_tau_k * step_size[tid];
            alpha_scatti[tid] = alpha_scattf;

            double alpha_absf = cuda_radiation::alpha_inv_abs(
                nu[tid], fluid_params.theta_e, fluid_params.n_e, fluid_params.b, theta[tid], tables.k2);
            d_tau_abs[tid] = 0.5 * (alpha_absi[tid] + alpha_absf) * d_tau_k * step_size[tid];
            alpha_absi[tid] = alpha_absf;

            double bf = bias_func(bias_norm, fluid_params.theta_e, photon.w[tid]);
            bias[tid] = 0.5 * (bi[tid] + bf);
            bi[tid] = bf;
        }
    }
}

static __global__ void interact_photon_2(curandStatePhilox4_32_10_t *rng_state,
                                         const struct harm::Header *header,
                                         const struct cuda_harm::Data data,
                                         const struct harm::Units *units,
                                         const struct cuda_harm::Tables tables,
                                         struct PhotonArray photon,
                                         enum PhotonState *photon_state,
                                         bool *interact_cond,
                                         bool *scatter_cond,
                                         struct PhotonArray photon_2,
                                         struct PhotonArray photon_p,
                                         struct harm::FluidParams *fluid_params,
                                         double *g_cov,
                                         double *step_size,
                                         double bias_norm,
                                         double *theta,
                                         double *nu,
                                         double *alpha_scatti,
                                         double *alpha_absi,
                                         double *bi,
                                         double *d_tau_scatt,
                                         double *d_tau_abs,
                                         double *bias) {
    double g_cov_[consts::cuda::block_dim][consts::n_dim][consts::n_dim];
    for (int tid = threadIdx.x + blockIdx.x * blockDim.x; tid < n_photons; tid += blockDim.x * gridDim.x) {
        scatter_cond[tid] = false;

        if (photon_state[tid] != PhotonState::Initialized || !interact_cond[tid]) {
            continue;
        }

        double x1 = -log(curand_uniform(&rng_state[tid]));

        photon_p.w[tid] = photon.w[tid] / bias[tid];

        if (bias[tid] * d_tau_scatt[tid] > x1 && photon_p.w[tid] > consts::weight_min) {
            double frac = x1 / (bias[tid] * d_tau_scatt[tid]);

            /* apply absorption until scattering event */
            d_tau_abs[tid] *= frac;

            if (d_tau_abs[tid] > 100) {
                /* this photon has been absorbed before scattering */
                photon_state[tid] = PhotonState::Empty;
                continue;
            }

            d_tau_scatt[tid] *= frac;

            double d_tau = d_tau_abs[tid] + d_tau_scatt[tid];

            if (d_tau_abs[tid] < 1.0e-3) {
                photon.w[tid] *= (1.0 - d_tau / 24.0 * (24.0 - d_tau * (12.0 - d_tau * (4.0 - d_tau))));
            } else {
                photon.w[tid] *= exp(-d_tau);
            }

            struct photon::Photon p = {
                .x = {photon_2.x[0][tid], photon_2.x[1][tid], photon_2.x[2][tid], photon_2.x[3][tid]},
                .k = {photon_2.k[0][tid], photon_2.k[1][tid], photon_2.k[2][tid], photon_2.k[3][tid]},
                .dkdlam = {photon_2.dkdlam[0][tid],
                           photon_2.dkdlam[1][tid],
                           photon_2.dkdlam[2][tid],
                           photon_2.dkdlam[3][tid]},
                .e_0_s = photon_2.e_0_s[tid],
            };

            push_photon(header, &p, step_size[tid] * frac);

#pragma unroll
            for (int i = 0; i < consts::n_dim; ++i) {
                photon.x[i][tid] = p.x[i];
                photon.k[i][tid] = p.k[i];
                photon.dkdlam[i][tid] = p.dkdlam[i];
            }
            photon.e_0_s[tid] = p.e_0_s;

            const double photon_x[4] = {photon.x[0][tid], photon.x[1][tid], photon.x[2][tid], photon.x[3][tid]};
            const double photon_k[4] = {photon.k[0][tid], photon.k[1][tid], photon.k[2][tid], photon.k[3][tid]};

            cuda_harm::gcov_func(header, photon_x, g_cov_[threadIdx.x]);

            harm::FluidParams fluid_params_ = cuda_harm::get_fluid_params(header,
                                                                          units,
                                                                          data.k_rho,
                                                                          data.u,
                                                                          data.u_1,
                                                                          data.u_2,
                                                                          data.u_3,
                                                                          data.b_1,
                                                                          data.b_2,
                                                                          data.b_3,
                                                                          photon_x,
                                                                          g_cov_[threadIdx.x]);

            if (fluid_params_.n_e > 0.0) {
                scatter_cond[tid] = true;

                if (photon.k[0][tid] > 1.0e5 || photon.k[0][tid] < 0.0 || isnan(photon.k[0][tid]) ||
                    isnan(photon.k[1][tid]) || isnan(photon.k[3][tid])) {
                    photon.k[0][tid] = fabs(photon.k[0][tid]);
                    photon.w[tid] = 0.0;
                }

                if (photon.w[tid] < 1.0e-100) {
                    /* must have been a problem popping k back onto light cone */
                    photon_state[tid] = PhotonState::Empty;
                    continue;
                }

#pragma unroll
                for (int i = 0; i < consts::n_dim; ++i) {
#pragma unroll
                    for (int j = 0; j < consts::n_dim; ++j) {
                        g_cov[tid * consts::n_dim * consts::n_dim + i * consts::n_dim + j] = g_cov_[threadIdx.x][i][j];
                    }
                }
                fluid_params[tid] = fluid_params_;
            }

            theta[tid] = cuda_radiation::bk_angle(
                photon_x, photon_k, fluid_params_.u_cov, fluid_params_.b_cov, fluid_params_.b, units->b_unit);
            nu[tid] = cuda_radiation::fluid_nu(photon_x, photon_k, fluid_params_.u_cov);

            if (nu[tid] < 0.0) {
                alpha_scatti[tid] = 0.0;
                alpha_absi[tid] = 0.0;
            } else {
                alpha_scatti[tid] = cuda_radiation::alpha_inv_scatt(
                    nu[tid], fluid_params_.theta_e, fluid_params_.n_e, tables.hotcross_table);
                alpha_absi[tid] = cuda_radiation::alpha_inv_abs(
                    nu[tid], fluid_params_.theta_e, fluid_params_.n_e, fluid_params_.b, theta[tid], tables.k2);
            }
            bi[tid] = bias_func(bias_norm, fluid_params_.theta_e, photon.w[tid]);

        } else {
            if (d_tau_abs[tid] > 100) {
                /* this photon has been absorbed */
                photon_state[tid] = PhotonState::Empty;
                continue;
            }

            double d_tau = d_tau_abs[tid] + d_tau_scatt[tid];
            if (d_tau < 1.0e-3) {
                photon.w[tid] *= (1.0 - d_tau / 24.0 * (24.0 - d_tau * (12.0 - d_tau * (4.0 - d_tau))));
            } else {
                photon.w[tid] *= exp(-d_tau);
            }
        }

        photon.tau_abs[tid] += d_tau_abs[tid];
        photon.tau_scatt[tid] += d_tau_scatt[tid];
    }
}

static __global__ void scatter_super_photon(curandStatePhilox4_32_10_t *rng_state,
                                            const struct harm::Units *units,
                                            struct PhotonArray photon,
                                            enum PhotonState *photon_state,
                                            bool *scatter_cond,
                                            struct PhotonArray photon_p,
                                            struct harm::FluidParams *fluid_params,
                                            double *g_cov) {
    __shared__ double g_cov_[consts::cuda::block_dim][consts::n_dim][consts::n_dim];

    for (int tid = threadIdx.x + blockIdx.x * blockDim.x; tid < n_photons; tid += blockDim.x * gridDim.x) {
        if ((photon_state[tid] != PhotonState::Initialized && photon_state[tid] != PhotonState::Tracked) ||
            !scatter_cond[tid]) {
            continue;
        }

#pragma unroll
        for (int i = 0; i < consts::n_dim; ++i) {
#pragma unroll
            for (int j = 0; j < consts::n_dim; ++j) {
                g_cov_[threadIdx.x][i][j] = g_cov[tid * consts::n_dim * consts::n_dim + i * consts::n_dim + j];
            }
        }

        double b_hat_con[consts::n_dim];

        if (fluid_params[tid].b > 0.0) {
            for (int i = 0; i < consts::n_dim; ++i) {
                b_hat_con[i] = fluid_params[tid].b_con[i] / (fluid_params[tid].b / units->b_unit);
            }
        } else {
            for (int i = 0; i < consts::n_dim; ++i) {
                b_hat_con[i] = 0.0;
            }
            b_hat_con[1] = 1.0;
        }

        double e_con[consts::n_dim][consts::n_dim];
        double e_cov[consts::n_dim][consts::n_dim];

        /* local tetrad */
        cuda_tetrads::make_tetrad(fluid_params[tid].u_con, b_hat_con, g_cov_[threadIdx.x], e_con, e_cov);

        const double photon_k[4] = {photon.k[0][tid], photon.k[1][tid], photon.k[2][tid], photon.k[3][tid]};
        double k_tetrad[consts::n_dim];

        cuda_tetrads::coordinate_to_tetrad(e_cov, photon_k, k_tetrad);

        if (k_tetrad[0] > 1.0e5 || k_tetrad[0] < 0.0 || isnan(k_tetrad[1])) {
            scatter_cond[tid] = false;
            continue;
        }

        double p[consts::n_dim];
        cuda_proba::sample_electron_distr_p(&rng_state[tid], k_tetrad, p, fluid_params[tid].theta_e);

        double k_tetrad_p[consts::n_dim];
        sample_scattered_photon(&rng_state[tid], k_tetrad, p, k_tetrad_p);

        double photon_p_k[4];

        cuda_tetrads::tetrad_to_coordinate(e_con, k_tetrad_p, photon_p_k);

#pragma unroll
        for (int i = 0; i < consts::n_dim; ++i) {
            photon_p.k[i][tid] = photon_p_k[i];
        }

        if (isnan(photon_p.k[1][tid])) {
            photon_p.w[tid] = 0.0;
            scatter_cond[tid] = false;
            continue;
        }

        double tmp_k[consts::n_dim];
        k_tetrad_p[0] *= -1.0;
        cuda_tetrads::tetrad_to_coordinate(e_cov, k_tetrad_p, tmp_k);

        photon_p.e[tid] = -tmp_k[0];
        photon_p.e_0_s[tid] = -tmp_k[0];
        photon_p.l[tid] = tmp_k[3];
        photon_p.tau_abs[tid] = 0.0;
        photon_p.tau_scatt[tid] = 0.0;
        photon_p.b_0[tid] = fluid_params[tid].b;

        photon_p.x1i[tid] = photon.x[1][tid];
        photon_p.x2i[tid] = photon.x[2][tid];
        photon_p.x[0][tid] = photon.x[0][tid];
        photon_p.x[1][tid] = photon.x[1][tid];
        photon_p.x[2][tid] = photon.x[2][tid];
        photon_p.x[3][tid] = photon.x[3][tid];

        photon_p.n_e_0[tid] = photon.n_e_0[tid];
        photon_p.theta_e_0[tid] = photon.theta_e_0[tid];
        photon_p.e_0[tid] = photon.e_0[tid];
        photon_p.n_scatt[tid] = photon.n_scatt[tid] + 1;
    }
}

static __global__ void incr_check_n_step(int *n_step, enum PhotonState *photon_state) {
    for (int tid = threadIdx.x + blockIdx.x * blockDim.x; tid < n_photons; tid += blockDim.x * gridDim.x) {
        if (photon_state[tid] != PhotonState::Initialized) {
            continue;
        }
        ++n_step[tid];

        if (n_step[tid] > consts::max_n_step) {
            photon_state[tid] = PhotonState::Empty;
        }
    }
}

static __global__ void record_super_photon(const struct harm::Header *header,
                                           struct PhotonArray photon,
                                           enum PhotonState *photon_state,
                                           int *n_step,
                                           struct harm::Spectrum *spectrum) {
    const double x1_max = log(consts::r_max);
    const double l_e_0 = log(1.0e-12);

    for (int tid = threadIdx.x + blockIdx.x * blockDim.x; tid < n_photons; tid += blockDim.x * gridDim.x) {
        if (photon_state[tid] != PhotonState::Tracked) {
            continue;
        }

        photon_state[tid] = PhotonState::Empty;

        /* record criterion */
        if (photon.x[1][tid] <= x1_max || isnan(photon.w[tid]) || isnan(photon.e[tid])) {
            continue;
        }

        atomic_max_double(&dev_max_tau_scatt, photon.tau_scatt[tid]);

        double dx2 = (header->x_stop[2] - header->x_start[2]) / (2.0 * consts::n_th_bins);
        int ix2;
        if (photon.x[2][tid] < 0.5 * (header->x_start[2] + header->x_stop[2])) {
            ix2 = static_cast<int>(photon.x[2][tid] / dx2);
        } else {
            ix2 = static_cast<int>((header->x_stop[2] - photon.x[2][tid]) / dx2);
        }

        if (ix2 < 0 || ix2 >= consts::n_th_bins) {
            continue;
        }

        double l_e = log(photon.e[tid]);
        int i_e = static_cast<int>((l_e - l_e_0) / consts::spectrum::d_l_e + 2.5) - 2;

        if (i_e < 0 || i_e >= consts::n_e_bins) {
            continue;
        }

        atomicAdd(&dev_n_super_photon_recorded, 1);
        atomicAdd(&dev_n_super_photon_scatt, photon.n_scatt[tid]);

        /* sum in photon */
        const int idx = ix2 * consts::n_e_bins + i_e;

        /* TODO: optimize it using reduction */
        atomicAdd(&spectrum[idx].dn_dle, photon.w[tid]);
        atomicAdd(&spectrum[idx].de_dle, photon.w[tid] * photon.e[tid]);
        atomicAdd(&spectrum[idx].tau_abs, photon.w[tid] * photon.tau_abs[tid]);
        atomicAdd(&spectrum[idx].tau_scatt, photon.w[tid] * photon.tau_scatt[tid]);
        atomicAdd(&spectrum[idx].x1i_av, photon.w[tid] * photon.x1i[tid]);
        atomicAdd(&spectrum[idx].x2i_sq, photon.w[tid] * (photon.x2i[tid] * photon.x2i[tid]));
        atomicAdd(&spectrum[idx].x3f_sq, photon.w[tid] * (photon.x[3][tid] * photon.x[3][tid]));
        atomicAdd(&spectrum[idx].ne_0, photon.w[tid] * (photon.n_e_0[tid]));
        atomicAdd(&spectrum[idx].b_0, photon.w[tid] * (photon.b_0[tid]));
        atomicAdd(&spectrum[idx].theta_e_0, photon.w[tid] * (photon.theta_e_0[tid]));
        atomicAdd(&spectrum[idx].nscatt, photon.n_scatt[tid]);
        atomicAdd(&spectrum[idx].nph, 1.0);
    }
}

static __device__ double bias_func(double bias_norm, double t_e, double w) {
    double max = 0.5 * w / consts::weight_min;
    double avg_num_scatt = dev_n_super_photon_scatt / (1.0 * dev_n_super_photon_recorded + 1.0);
    double bias = 100.0 * t_e * t_e / (bias_norm * dev_max_tau_scatt * (avg_num_scatt + 2.0));

    if (bias < consts::tp_over_te) {
        bias = consts::tp_over_te;
    }
    if (bias > max) {
        bias = max;
    }

    return bias / consts::tp_over_te;
}

static __device__ void init_dkdlam(const struct harm::Header *header,
                                   const double (&x)[consts::n_dim],
                                   const double (&k_con)[consts::n_dim],
                                   double (&d_k)[consts::n_dim]) {
    double lconn[lconn_flat_len];

    get_connection(header, x, lconn);

#pragma unroll
    for (int i = 0; i < consts::n_dim; ++i) {
        d_k[i] =
            -2.0 * (k_con[0] * (lconn[lconn_flat_idx(i, 0, 1)] * k_con[1] + lconn[lconn_flat_idx(i, 0, 2)] * k_con[2] +
                                lconn[lconn_flat_idx(i, 0, 3)] * k_con[3]) +
                    k_con[1] * (lconn[lconn_flat_idx(i, 1, 2)] * k_con[2] + lconn[lconn_flat_idx(i, 1, 3)] * k_con[3]) +
                    lconn[lconn_flat_idx(i, 2, 3)] * k_con[2] * k_con[3]);

        d_k[i] -= (lconn[lconn_flat_idx(i, 0, 0)] * k_con[0] * k_con[0] +
                   lconn[lconn_flat_idx(i, 1, 1)] * k_con[1] * k_con[1] +
                   lconn[lconn_flat_idx(i, 2, 2)] * k_con[2] * k_con[2] +
                   lconn[lconn_flat_idx(i, 3, 3)] * k_con[3] * k_con[3]);
    }
}

static __device__ void
get_connection(const struct harm::Header *header, const double (&x)[consts::n_dim], double (&lconn)[lconn_flat_len]) {
    const double r1 = exp(x[1]);
    const double r2 = r1 * r1;
    const double r3 = r2 * r1;
    const double r4 = r3 * r1;

    double s_x;
    double c_x;
    sincos(2.0 * CUDART_PI * x[2], &s_x, &c_x);

    const double th = CUDART_PI * x[2] + 0.5 * (1.0 - header->h_slope) * s_x;
    const double dthdx2 = CUDART_PI * (1.0 + (1.0 - header->h_slope) * c_x);
    const double d2thdx22 = -2.0 * CUDART_PI * CUDART_PI * (1.0 - header->h_slope) * s_x;
    const double dthdx22 = dthdx2 * dthdx2;

    double sth;
    double cth;
    sincos(th, &sth, &cth);

    const double sth2 = sth * sth;
    const double r1sth2 = r1 * sth2;
    const double sth4 = sth2 * sth2;
    const double cth2 = cth * cth;
    const double cth4 = cth2 * cth2;
    const double s2th = 2.0 * sth * cth;
    const double c2th = 2.0 * cth2 - 1.0;

    const double a = header->a;
    const double a2 = a * a;
    const double a3 = a2 * a;
    const double a4 = a3 * a;
    const double a2sth2 = a2 * sth2;
    const double a2cth2 = a2 * cth2;
    const double a4cth4 = a4 * cth4;

    const double rho2 = r2 + a2cth2;
    const double rho22 = rho2 * rho2;
    const double rho23 = rho22 * rho2;
    const double irho2 = 1.0 / rho2;
    const double irho22 = irho2 * irho2;
    const double irho23 = irho22 * irho2;
    const double irho23_dthdx2 = irho23 / dthdx2;

    const double fac1 = r2 - a2cth2;
    const double fac1_rho23 = fac1 * irho23;
    const double fac2 = a2 + 2.0 * r2 + a2 * c2th;
    const double fac3 = a2 + r1 * (-2.0 + r1);

    lconn[lconn_flat_idx(0, 0, 0)] = 2.0 * r1 * fac1_rho23;
    lconn[lconn_flat_idx(0, 0, 1)] = r1 * (2.0 * r1 + rho2) * fac1_rho23;
    lconn[lconn_flat_idx(0, 0, 2)] = -a2 * r1 * s2th * dthdx2 * irho22;
    lconn[lconn_flat_idx(0, 0, 3)] = -2.0 * a * r1sth2 * fac1_rho23;

    /* lconn[0][1][0] = lconn[0][0][1]; */
    lconn[lconn_flat_idx(0, 1, 1)] = 2.0 * r2 * (r4 + r1 * fac1 - a4cth4) * irho23;
    lconn[lconn_flat_idx(0, 1, 2)] = -a2 * r2 * s2th * dthdx2 * irho22;
    lconn[lconn_flat_idx(0, 1, 3)] = a * r1 * (-r1 * (r3 + 2.0 * fac1) + a4cth4) * sth2 * irho23;

    /* lconn[0][2][0] = lconn[0][0][2]; */
    /* lconn[0][2][1] = lconn[0][1][2]; */
    lconn[lconn_flat_idx(0, 2, 2)] = -2.0 * r2 * dthdx22 * irho2;
    lconn[lconn_flat_idx(0, 2, 3)] = a3 * r1sth2 * s2th * dthdx2 * irho22;

    /* lconn[0][3][0] = lconn[0][0][3]; */
    /* lconn[0][3][1] = lconn[0][1][3]; */
    /* lconn[0][3][2] = lconn[0][2][3]; */
    lconn[lconn_flat_idx(0, 3, 3)] = 2.0 * r1sth2 * (-r1 * rho22 + a2sth2 * fac1) * irho23;

    lconn[lconn_flat_idx(1, 0, 0)] = fac3 * fac1 / (r1 * rho23);
    lconn[lconn_flat_idx(1, 0, 1)] = fac1 * (-2.0 * r1 + a2sth2) * irho23;
    lconn[lconn_flat_idx(1, 0, 2)] = 0.0;
    lconn[lconn_flat_idx(1, 0, 3)] = -a * sth2 * fac3 * fac1 / (r1 * rho23);

    /* lconn[1][1][0] = lconn[1][0][1]; */
    lconn[lconn_flat_idx(1, 1, 1)] =
        (r4 * (-2.0 + r1) * (1.0 + r1) + a2 * (a2 * r1 * (1.0 + 3.0 * r1) * cth4 + a4cth4 * cth2 + r3 * sth2 +
                                               r1 * cth2 * (2.0 * r1 + 3.0 * r3 - a2sth2))) *
        irho23;
    lconn[lconn_flat_idx(1, 1, 2)] = -a2 * dthdx2 * s2th / fac2;
    lconn[lconn_flat_idx(1, 1, 3)] =
        a * sth2 * (a4 * r1 * cth4 + r2 * (2.0 * r1 + r3 - a2sth2) + a2cth2 * (2.0 * r1 * (-1.0 + r2) + a2sth2)) *
        irho23;

    /* lconn[1][2][0] = lconn[1][0][2]; */
    /* lconn[1][2][1] = lconn[1][1][2]; */
    lconn[lconn_flat_idx(1, 2, 2)] = -fac3 * dthdx22 * irho2;
    lconn[lconn_flat_idx(1, 2, 3)] = 0.0;

    /* lconn[1][3][0] = lconn[1][0][3]; */
    /* lconn[1][3][1] = lconn[1][1][3]; */
    /* lconn[1][3][2] = lconn[1][2][3]; */
    lconn[lconn_flat_idx(1, 3, 3)] = -fac3 * sth2 * (r1 * rho22 - a2 * fac1 * sth2) / (r1 * rho23);

    lconn[lconn_flat_idx(2, 0, 0)] = -a2 * r1 * s2th * irho23_dthdx2;
    lconn[lconn_flat_idx(2, 0, 1)] = r1 * lconn[lconn_flat_idx(2, 0, 0)];
    lconn[lconn_flat_idx(2, 0, 2)] = 0.0;
    lconn[lconn_flat_idx(2, 0, 3)] = a * r1 * (a2 + r2) * s2th * irho23_dthdx2;

    /* lconn[2][1][0] = lconn[2][0][1]; */
    lconn[lconn_flat_idx(2, 1, 1)] = r2 * lconn[lconn_flat_idx(2, 0, 0)];
    lconn[lconn_flat_idx(2, 1, 2)] = r2 * irho2;
    lconn[lconn_flat_idx(2, 1, 3)] =
        (a * r1 * cth * sth * (r3 * (2.0 + r1) + a2 * (2.0 * r1 * (1.0 + r1) * cth2 + a2 * cth4 + 2.0 * r1sth2))) *
        irho23_dthdx2;

    /* lconn[2][2][0] = lconn[2][0][2]; */
    /* lconn[2][2][1] = lconn[2][1][2]; */
    lconn[lconn_flat_idx(2, 2, 2)] = -a2 * cth * sth * dthdx2 * irho2 + d2thdx22 / dthdx2;
    lconn[lconn_flat_idx(2, 2, 3)] = 0.0;

    /* lconn[2][3][0] = lconn[2][0][3]; */
    /* lconn[2][3][1] = lconn[2][1][3]; */
    /* lconn[2][3][2] = lconn[2][2][3]; */
    lconn[lconn_flat_idx(2, 3, 3)] =
        -cth * sth * (rho23 + a2sth2 * rho2 * (r1 * (4.0 + r1) + a2cth2) + 2.0 * r1 * a4 * sth4) * irho23_dthdx2;

    lconn[lconn_flat_idx(3, 0, 0)] = a * fac1_rho23;
    lconn[lconn_flat_idx(3, 0, 1)] = r1 * lconn[lconn_flat_idx(3, 0, 0)];
    lconn[lconn_flat_idx(3, 0, 2)] = -2.0 * a * r1 * cth * dthdx2 / (sth * rho22);
    lconn[lconn_flat_idx(3, 0, 3)] = -a2sth2 * fac1_rho23;

    /* lconn[3][1][0] = lconn[3][0][1]; */
    lconn[lconn_flat_idx(3, 1, 1)] = a * r2 * fac1_rho23;
    lconn[lconn_flat_idx(3, 1, 2)] =
        -2 * a * r1 * (a2 + 2.0 * r1 * (2.0 + r1) + a2 * c2th) * cth * dthdx2 / (sth * fac2 * fac2);
    lconn[lconn_flat_idx(3, 1, 3)] = r1 * (r1 * rho22 - a2sth2 * fac1) * irho23;

    /* lconn[3][2][0] = lconn[3][0][2]; */
    /* lconn[3][2][1] = lconn[3][1][2]; */
    lconn[lconn_flat_idx(3, 2, 2)] = -a * r1 * dthdx22 * irho2;
    lconn[lconn_flat_idx(3, 2, 3)] = dthdx2 * (0.25 * fac2 * fac2 * cth / sth + a2 * r1 * s2th) * irho22;

    /* lconn[3][3][0] = lconn[3][0][3]; */
    /* lconn[3][3][1] = lconn[3][1][3]; */
    /* lconn[3][3][2] = lconn[3][2][3]; */
    lconn[lconn_flat_idx(3, 3, 3)] = (-a * r1sth2 * rho22 + a3 * sth4 * fac1) * irho23;
}

static __device__ void push_photon(const struct harm::Header *header, struct photon::Photon *photon, double dl) {
    if (photon->x[1] < header->x_start[1]) {
        return;
    }

    double dl_stack[8] = {dl};
    int depth_stack[8] = {0};
    int n = 0;

    double x_cpy[consts::n_dim];
    double k_cpy[consts::n_dim];
    double dk_cpy[consts::n_dim];

    while (n >= 0) {
#pragma unroll
        for (int i = 0; i < consts::n_dim; ++i) {
            x_cpy[i] = photon->x[i];
            k_cpy[i] = photon->k[i];
            dk_cpy[i] = photon->dkdlam[i];
        }

        auto [e_1, err, err_e] = push_photon_step(header, photon, dl_stack[n]);

        if (depth_stack[n] < 7 && (err_e > 1.0e-4 || err > consts::e_tol || !isfinite(err))) {
#pragma unroll
            for (int i = 0; i < consts::n_dim; ++i) {
                photon->x[i] = x_cpy[i];
                photon->k[i] = k_cpy[i];
                photon->dkdlam[i] = dk_cpy[i];
            }
            dl_stack[n] = dl_stack[n] / 2.0;
            dl_stack[n + 1] = dl_stack[n];
            depth_stack[n] = depth_stack[n] + 1;
            depth_stack[n + 1] = depth_stack[n];
        } else {
            photon->e_0_s = e_1;
            --n;
        }
    }
}

static __device__ std::tuple<double, double, double>
push_photon_step(const struct harm::Header *header, struct photon::Photon *photon, double dl) {
    const double dl_2 = 0.5 * dl;
    double k[consts::n_dim];

#pragma unroll
    for (int i = 0; i < consts::n_dim; ++i) {
        double dk = photon->dkdlam[i] * dl_2;
        photon->k[i] += dk;
        k[i] = photon->k[i] + dk;
        photon->x[i] += photon->k[i] * dl;
    }

    double lconn[lconn_flat_len];

    get_connection(header, photon->x, lconn);

    double err;
    int iter = 0;

    do {
        ++iter;

        err = 0.0;

#pragma unroll
        for (int i = 0; i < consts::n_dim; ++i) {
            photon->dkdlam[i] =
                -2.0 * (k[0] * (lconn[lconn_flat_idx(i, 0, 1)] * k[1] + lconn[lconn_flat_idx(i, 0, 2)] * k[2] +
                                lconn[lconn_flat_idx(i, 0, 3)] * k[3]) +
                        k[1] * (lconn[lconn_flat_idx(i, 1, 2)] * k[2] + lconn[lconn_flat_idx(i, 1, 3)] * k[3]) +
                        lconn[lconn_flat_idx(i, 2, 3)] * k[2] * k[3]);
            photon->dkdlam[i] -=
                (lconn[lconn_flat_idx(i, 0, 0)] * k[0] * k[0] + lconn[lconn_flat_idx(i, 1, 1)] * k[1] * k[1] +
                 lconn[lconn_flat_idx(i, 2, 2)] * k[2] * k[2] + lconn[lconn_flat_idx(i, 3, 3)] * k[3] * k[3]);

            double old_k = k[i];
            k[i] = fma(dl_2, photon->dkdlam[i], photon->k[i]);
            err += fabs((old_k - k[i]) / (k[i] + consts::eps));
        }
    } while (err > consts::e_tol && iter < consts::max_iter);

#pragma unroll
    for (int i = 0; i < consts::n_dim; ++i) {
        photon->k[i] = k[i];
    }

    double g_cov_0[consts::n_dim];

    cuda_harm::gcov_0_func(header, photon->x, g_cov_0);

    /* clang-format off */
    double e_1 = -(
        photon->k[0] * g_cov_0[0]
      + photon->k[1] * g_cov_0[1]
      + photon->k[2] * g_cov_0[2]
      + photon->k[3] * g_cov_0[3]);
    /* clang-format on */

    double err_e = fabs((e_1 - photon->e_0_s) / photon->e_0_s);

    return {e_1, err, err_e};
}

static __device__ void sample_scattered_photon(curandStatePhilox4_32_10_t *rng_state,
                                               const double (&k)[consts::n_dim],
                                               double (&p)[consts::n_dim],
                                               double (&kp)[consts::n_dim]) {
    double ke[consts::n_dim];

    boost(k, p, ke);

    double k0p;
    double c_th;

    if (ke[0] > 1.0e-4) {
        k0p = cuda_proba::sample_klein_nishina(rng_state, ke[0]);
        c_th = 1.0 - 1.0 / k0p + 1.0 / ke[0];
    } else {
        k0p = ke[0];
        c_th = cuda_proba::sample_thomson(rng_state);
    }
    double s_th = sqrt(abs(1.0 - c_th * c_th));

    double v0x = ke[1] / ke[0];
    double v0y = ke[2] / ke[0];
    double v0z = ke[3] / ke[0];

    double n0x;
    double n0y;
    double n0z;
    cuda_proba::sample_rand_dir(rng_state, &n0x, &n0y, &n0z);

    double n0dotv0 = v0x * n0x + v0y * n0y + v0z * n0z;

    /* unit vector 2 */
    double v1x = n0x - (n0dotv0)*v0x;
    double v1y = n0y - (n0dotv0)*v0y;
    double v1z = n0z - (n0dotv0)*v0z;
    double v1 = sqrt(v1x * v1x + v1y * v1y + v1z * v1z);
    v1x /= v1;
    v1y /= v1;
    v1z /= v1;

    /* find one more unit vector using cross product;
       this guy is automatically normalized */
    double v2x = v0y * v1z - v0z * v1y;
    double v2y = v0z * v1x - v0x * v1z;
    double v2z = v0x * v1y - v0y * v1x;

    /* now resolve new momentum vector along unit vectors */
    /* create a four-vector $p$ */
    /* solve for orientation of scattered photon */

    /* find phi for new photon */
    double phi = 2.0 * CUDART_PI * curand_uniform(rng_state);
    double s_phi = sin(phi);
    double c_phi = cos(phi);

    p[1] *= -1.;
    p[2] *= -1.;
    p[3] *= -1.;

    double dir1 = c_th * v0x + s_th * (c_phi * v1x + s_phi * v2x);
    double dir2 = c_th * v0y + s_th * (c_phi * v1y + s_phi * v2y);
    double dir3 = c_th * v0z + s_th * (c_phi * v1z + s_phi * v2z);

    double kpe[consts::n_dim] = {
        k0p,
        k0p * dir1,
        k0p * dir2,
        k0p * dir3,
    };

    /* transform k back to lab frame */
    boost(kpe, p, kp);
}

static __device__ void
boost(const double (&v)[consts::n_dim], const double (&u)[consts::n_dim], double (&vp)[consts::n_dim]) {
    double g = u[0];
    double v_ = sqrt(abs(1.0 - 1.0 / (g * g)));
    double n1 = u[1] / (g * v_ + consts::eps);
    double n2 = u[2] / (g * v_ + consts::eps);
    double n3 = u[3] / (g * v_ + consts::eps);
    double gm1 = g - 1.0;

    /* general Lorentz boost into frame u from lab frame */
    vp[0] = u[0] * v[0] - u[1] * v[1] - u[2] * v[2] - u[3] * v[3];
    vp[1] = -u[1] * v[0] + (1.0 + n1 * n1 * gm1) * v[1] + n1 * n2 * gm1 * v[2] + n1 * n3 * gm1 * v[3];
    vp[2] = -u[2] * v[0] + n2 * n1 * gm1 * v[1] + (1.0 + n2 * n2 * gm1) * v[2] + n2 * n3 * gm1 * v[3];
    vp[3] = -u[3] * v[0] + n3 * n1 * gm1 * v[1] + n3 * n2 * gm1 * v[2] + (1.0 + n3 * n3 * gm1) * v[3];
}

static __device__ double atomic_max_double(double *addr, double val) {
    /* NOLINTBEGIN */
    unsigned long long int *addr_as_ull = (unsigned long long int *)addr;

    unsigned long long int old = *addr_as_ull;
    unsigned long long int assumed;
    /* NOLINTEND */

    do {
        assumed = old;
        double old_val = __longlong_as_double(assumed);
        if (old_val >= val) {
            break; // already bigger
        }
        old = atomicCAS(addr_as_ull, assumed, __double_as_longlong(val));
    } while (assumed != old);

    return __longlong_as_double(old);
}

}; /* namespace cuda_super_photon */
