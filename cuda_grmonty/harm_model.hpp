/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <semaphore>
#include <string>
#include <tuple>

#include "cuda_grmonty/consts.hpp"
#include "cuda_grmonty/harm_data.hpp"
#include "cuda_grmonty/ndarray.hpp"
#include "cuda_grmonty/photon.hpp"
#include "cuda_grmonty/utils.hpp"

namespace harm {

/**
 * @class HARMModel
 * @brief Monte Carlo photon transport model using HARM simulation data.
 *
 * This class loads magnetohydrodynamic (MHD) simulation output from HARM, initializes the physical and numerical
 * parameters, and runs a Monte Carlo radiative transfer simulation of photons moving through the plasma. It supports
 * photon propagation, scattering, absorption, and spectrum accumulation.
 */
class HARMModel {
public:
    /**
     * @brief Construct a new HARMModel object.
     *
     * @param photon_n  Number of photons to simulate.
     * @param mass_unit Mass unit used for normalization/scaling.
     */
    explicit HARMModel(int photon_n, double mass_unit);

    HARMModel(const HARMModel &) = delete;

    HARMModel &operator=(const HARMModel &) = delete;

    /**
     * @brief Reads HARM data from file.
     *
     * @param filepath Path to HARM dump.
     */
    void read_file(std::string filepath);

    /**
     * @brief Initializes HARM model.
     *
     * Prepares internal state and data structures for running the simulation.
     */
    void init();

    /**
     * @brief Runs the photon transport simulation.
     *
     * Executes the main Monte Carlo loop that propagates photons, applies interactions, and accumulates results into
     * the spectrum.
     */
    void run_simulation();

    /**
     * @brief Writes the computed photon spectrum to a file.
     *
     * @param filepath Output path for the spectrum data file.
     */
    void report_spectrum(std::string filepath);

    /**
     * @brief Getter for HARM header.
     *
     * @return Pointer to the header read from HARM file.
     */
    const struct Header *get_header() const { return &header_; }

    /**
     * @brief Getter for HARM data.
     *
     * @return Pointer to the data read from HARM file.
     */
    const struct Data *get_data() const { return &data_; }

private:
    /**
     * @brief Header containing global simulation parameters (grid extents, spin, etc.) read from the HARM dump.
     */
    struct Header header_;
    /**
     * @brief Fluid and metric field data loaded from the HARM dump.
     */
    struct Data data_;
    /**
     * @brief Unit conversion factors used for normalization of physical quantities (e.g., B-fields, density).
     */
    struct Units units_;

    /**
     * @brief Normalization factor for photon biasing during interaction sampling.
     */
    double bias_norm_;
    /**
     * @brief Event horizon radius in logarithmic coordinates.
     */
    double rh_;

    /**
     * @brief Total number of photons to simulate.
     */
    int photon_n_;
    /**
     * @brief Maximum scattering optical depth encountered so far.
     */
    double max_tau_scatt_;
    /**
     * @brief Scaling factor used in optical depth integration.
     */
    double d_tau_k_;
    /**
     * @brief Minimum allowed x1 coordinate (used to detect horizon crossing).
     */
    double x1_min_;

    /**
     * @brief Counter for total number of super-photons created.
     */
    uint64_t n_super_photon_created_ = 0;
    /**
     * @brief Counter for total number of scattering events among all super-photons.
     */
    uint64_t n_super_photon_scatt_ = 0;
    /**
     * @brief Counter for total number of recorded super-photons contributing to the output spectrum.
     */
    uint64_t n_super_photon_recorded_ = 0;

    /**
     * @brief Current zone index in x1 (radial/log-radius) direction.
     */
    int zone_x_1_ = 0;
    /**
     * @brief Current zone index in x2 (polar angle) direction. Initialized to -1 to indicate "no zone."
     */
    int zone_x_2_ = -1;

    /**
     * @brief Cached geometric quantities (metric, connections, etc.) at the photon position.
     */
    struct Geometry geometry_;
    /**
     * @brief Precomputed hot cross section lookup table for scattering.
     *
     * @details Indexed by frequency and angle bins. The dimensions are (n_w + 1) × (n_t + 1) where n_w is the number
     *          of frequency bins and n_t the number of scattering angle bins.
     */
    ndarray::NDArray<double, 2> hotcross_table_ =
        ndarray::NDArray<double, 2>({consts::hotcross::n_w + 1, consts::hotcross::n_t + 1});
    /**
     * @brief Tabulated electron distribution function.
     *
     * @details f_[i] contains samples of the normalized Maxwell–Jüttner distribution function at discrete energy
     *          points, used in importance sampling of electron energies.
     */
    std::array<double, consts::n_e_samp + 1> f_;
    /**
     * @brief Precomputed normalization constants for absorption coefficients.
     *
     * @details k2_[i] is computed from integrals of the electron distribution function, used to normalize synchrotron
     *          and Compton opacity contributions.
     */
    std::array<double, consts::n_e_samp + 1> k2_;
    /**
     * @brief Weighting function for photon sampling over electron energies.
     *
     * @details weight_[i] gives the probability weights for drawing electron energies from the tabulated distribution
     *          function, ensuring unbiased Monte Carlo sampling.
     */
    std::array<double, consts::n_e_samp + 1> weight_;
    /**
     * @brief Precomputed integrals for photon–electron interaction rates.
     *
     * @details nint_[i] stores integrals over the electron distribution for specific interaction kernels, used to
     *          accelerate rejection sampling and cross-section evaluation.
     */
    std::array<double, consts::nint + 1> nint_;
    /**
     * @brief Maximum differential number density per logarithmic frequency bin.
     *
     * @details dndlnu_max_[i] provides an upper bound used for rejection  sampling when drawing photon frequencies
     *          from the emissivity distribution.
     */
    std::array<double, consts::nint + 1> dndlnu_max_;

    /**
     * @brief Output spectrum histogram.
     *
     * @details Indexed by polar angle bin (ix2) and energy bin (i_e). Each bin accumulates weighted contributions from
     *          recorded photons, including number, energy, optical depths, and scattering statistics.
     */
    struct Spectrum spectrum_[consts::n_th_bins][consts::n_e_bins];

    /**
     * @brief Initializes metric-related geometric quantities.
     *
     * @details Precomputes the background spacetime geometry (metric tensors, Christoffel symbols, etc.) at grid
     *          points required for photon transport. Called during initialization.
     */
    void init_geometry();

    /**
     * @brief Initializes photon weight lookup table.
     *
     * @details Precomputes the electron distribution function weights used during Monte Carlo sampling of
     *          interactions, enabling efficient importance sampling.
     */
    void init_weight_table();

    /**
     * @brief Initializes nint lookup table.
     *
     * @details Precomputes integrals of the electron distribution used to accelerate evaluation of scattering and
     *          absorption rates.
     */
    void init_nint_table();

    /**
     * @brief Computes contravariant metric tensor g^μν at position x.
     *
     * @param[in] x      Coordinate vector (size = consts::n_dim).
     * @param[out] g_con Filled with contravariant metric tensor at x.
     */
    void gcon_func(const double (&x)[consts::n_dim], ndarray::NDArray<double, 2> &g_con) const;

    /**
     * @brief Computes covariant metric tensor g_μν at position x.
     *
     * @param[in] x      Coordinate vector (size = consts::n_dim).
     * @param[out] g_cov Filled with covariant metric tensor at x.
     */
    void gcov_func(const double (&x)[consts::n_dim], ndarray::NDArray<double, 2> &g_cov) const;

    /**
     * @brief Computes solid angle element ΔΩ between two polar angles.
     *
     * @param x2i Initial polar angle coordinate.
     * @param x2f Final polar angle coordinate.
     *
     * @return Solid angle subtended between x2i and x2f.
     */
    double d_omega_func(double x2i, double x2f) const;

    /**
     * @brief Retrieves fluid zone indices for given coordinates.
     *
     * @param x_1 Radial/log-radius zone index.
     * @param x_2 Polar angle zone index.
     *
     * @return FluidZone structure with fluid variables for this cell.
     */
    struct FluidZone get_fluid_zone(int x_1, int x_2) const;

    /**
     * @brief Interpolates fluid parameters at photon position.
     *
     * @param x     Photon position (size = consts::n_dim).
     * @param g_cov Covariant metric tensor at x.
     *
     * @return FluidParams structure with interpolated local fluid quantities (density, temperature, four-velocity,
     *         B-field).
     */
    struct FluidParams get_fluid_params(const double (&x)[consts::n_dim],
                                        const ndarray::NDArray<double, 2> &g_cov) const;

    /**
     * @brief Selects the next fluid zone for photon generation.
     *
     * @details Determines which zone should emit superphotons next and how many photons to generate based on emission
     *          weighting. Used during initialization and photon injection.
     *
     * @return A Zone structure describing the selected fluid cell and photon count to generate.
     */
    struct Zone get_zone();

    /**
     * @brief Samples a photon emitted from a given zone.
     *
     * @param zone Zone structure from which photon emission is to be sampled.
     *
     * @return Photon sampled with initial position, wavevector, and weight consistent with zone conditions.
     */
    struct photon::InitPhoton sample_zone_photon(struct Zone &zone);

    /**
     * @brief Computes linear interpolation weight for frequency nu.
     *
     * @param nu Frequency at which to interpolate.
     *
     * @return Interpolation weight used in constructing photon emission spectra.
     */
    double linear_interp_weight(double nu);

    /**
     * @brief Creates a new superphoton.
     *
     * @details Samples a photon from the current emission zone and determines whether it is accepted into the
     *          simulation. Also encodes whether a valid photon was produced.
     *
     * @return Tuple containing (Photon, success flag).
     */
    std::tuple<struct photon::InitPhoton, bool> make_super_photon();

    void make_super_photon_thread(unsigned int worked_id,
                                  utils::ConcurrentQueue<photon::InitPhoton> &photon_queue,
                                  utils::ConcurrentQueue<struct Zone> &work_queue,
                                  std::atomic<int> &n_rate);

    /**
     * @brief Asynchronously generates a new superphoton.
     *
     * @details Samples a photon and enqueues it into the provided photon queue, signaling the semaphore once complete.
     *
     * @param photon_queue Queue into which the photon is placed.
     * @param done_sem     Binary semaphore used to signal completion of photon generation.
     */
    void make_super_photon_async(utils::ConcurrentQueue<photon::InitPhoton> &photon_queue,
                                 std::binary_semaphore &done_sem);

    /**
     * @brief Propagates a superphoton through the simulation.
     *
     * @details Advances photon trajectory until it is absorbed, escapes, or reaches recording conditions. Handles
     *          scattering and absorption events.
     * @param[in,out] photon Photon to propagate.
     */
    void track_super_photon(struct photon::Photon &photon);

    /**
     * @brief Performs Compton scattering of a photon with the local fluid.
     *
     * @param[in,out] photon   Photon undergoing scattering (modified in place).
     * @param[out] photon_p    Secondary photon produced by scattering.
     * @param[in] fluid_params Local fluid parameters at scattering point.
     * @param[in] g_cov        Covariant metric tensor at photon position.
     * @param[in] b_unit       Magnetic field normalization constant.
     */
    void scatter_super_photon(struct photon::Photon &photon,
                              struct photon::Photon &photon_p,
                              const struct FluidParams &fluid_params,
                              const ndarray::NDArray<double, 2> &g_cov,
                              double b_unit) const;

    /**
     * @brief Samples a scattered photon’s momentum.
     *
     * @param[in] k   Incoming photon four-momentum.
     * @param[out] p  Electron four-momentum sampled from thermal distribution.
     * @param[out] kp Outgoing photon four-momentum after scattering.
     */
    void sample_scattered_photon(const double (&k)[consts::n_dim],
                                 double (&p)[consts::n_dim],
                                 double (&kp)[consts::n_dim]) const;

    /**
     * @brief Advances photon position along its trajectory.
     *
     * @param[in,out] photon Photon to update.
     * @param[in] dl         Affine parameter step size.
     * @param[in] n          Number of substeps for integration.
     */
    void push_photon(struct photon::Photon &photon, double dl, int n);

    /**
     * @brief Records photon contribution into the output spectrum.
     *
     * @param photon Photon to record.
     * @param n_step Current step index (used for diagnostics).
     */
    void record_super_photon(const struct photon::Photon &photon, int n_step);

    /**
     * @brief Initializes a fluid zone for photon generation.
     *
     * @param[in] x_1 Radial/log-radius index.
     * @param[in] x_2 Polar angle index.
     *
     * @return Tuple containing (cell volume element, zone emissivity weight).
     */
    std::tuple<double, double> init_zone(int x_1, int x_2) const;

    /**
     * @brief Computes bias factor for superphoton sampling.
     *
     * @param t_e Electron temperature in the zone.
     * @param w   Photon statistical weight.
     *
     * @return Bias factor applied during importance sampling.
     */
    double bias_func(double t_e, double w) const;

    /**
     * @brief Converts coordinates to grid indices.
     *
     * @param x Position in simulation coordinates.
     *
     * @return Tuple of (radial index, polar index, fractional radial offset, fractional polar offset).
     */
    std::tuple<int, int, double, double> x_to_ij(const double (&x)[consts::n_dim]) const;

    /**
     * @brief Computes Christoffel symbols (connection coefficients) at a given position.
     *
     * @param[in] x      Position in simulation coordinates.
     * @param[out] lconn Computed connection coefficients (Γ^μ_{νρ}).
     */
    void get_connection(const double (&x)[consts::n_dim], double (&lconn)[consts::n_dim][consts::n_dim][consts::n_dim]);

    /**
     * @brief Initializes geodesic derivatives of photon wavevector.
     *
     * @param[in] x     Position in simulation coordinates.
     * @param[in] k_con Contravariant wavevector components.
     * @param[out] d_k  Initial derivatives of wavevector with respect to affine parameter.
     */
    void
    init_dkdlam(const double (&x)[consts::n_dim], const double (&k_con)[consts::n_dim], double (&d_k)[consts::n_dim]);

    /**
     * @brief Determines whether photon propagation should stop.
     *
     * @param photon Photon being tracked.
     *
     * @return True if photon should be terminated (e.g. absorbed or escaped).
     */
    bool stop_criterion(struct photon::Photon &photon) const;

    /**
     * @brief Determines whether photon should be recorded into the spectrum.
     *
     * @param photon Photon being evaluated.
     *
     * @return True if photon meets recording conditions.
     */
    bool record_criterion(const struct photon::Photon &photon) const;

    /**
     * @brief Computes integration step size for photon trajectory.
     *
     * @param x Position in simulation coordinates.
     * @param k Photon wavevector.
     *
     * @return Step size for numerical integration of photon path.
     */
    double step_size(const double (&x)[consts::n_dim], const double (&k)[consts::n_dim]);

    /**
     * @brief Converts coordinates to Boyer–Lindquist system.
     *
     * @param x Position in simulation coordinates.
     *
     * @return Boyer–Lindquist coordinate structure.
     */
    struct BLCoord get_bl_coord(const double (&x)[consts::n_dim]) const;

    /**
     * @brief Retrieves coordinate position for a grid cell.
     *
     * @param[in] x_1 Radial/log-radius index.
     * @param[in] x_2 Polar index.
     * @param[out] x  Output coordinate in simulation coordinates.
     */
    void get_coord(int x_1, int x_2, double (&x)[consts::n_dim]) const;
};

}; /* namespace harm */
