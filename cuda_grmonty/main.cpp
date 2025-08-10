/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "cuda_grmonty/ndarray.hpp"
#include "spdlog/spdlog.h"

#include "cuda_grmonty/harm_model.hpp"
#include "cuda_grmonty/monty_rand.hpp"
#include "cuda_grmonty/parse_verbosity.hpp"

/* Define cli args */
ABSL_FLAG(int, photon_n, 5000000, "Estimate of photon number");
ABSL_FLAG(double, mass_unit, 4e19, "Mass unit");
ABSL_FLAG(std::string, harm_dump_path, "", "HARM dump file path");
ABSL_FLAG(std::string, spectrum_path, "", "Spectrum file path");
ABSL_FLAG(spdlog::level::level_enum, verbosity, spdlog::level::info, "Logging verbosity");

int main(int argc, char *argv[]) {
    absl::ParseCommandLine(argc, argv);

    auto photon_n = absl::GetFlag(FLAGS_photon_n);
    auto mass_unit = absl::GetFlag(FLAGS_mass_unit);
    auto harm_dump_path = absl::GetFlag(FLAGS_harm_dump_path);
    auto spectrum_path = absl::GetFlag(FLAGS_spectrum_path);
    auto verbosity = absl::GetFlag(FLAGS_verbosity);

    spdlog::set_level(verbosity);

    spdlog::info("Parameters:");
    spdlog::info("\tphoton_n: {}", photon_n);
    spdlog::info("\tmass_unit: {}", mass_unit);
    spdlog::info("\tharm_dump_path: {}", harm_dump_path);
    spdlog::info("\tspectrum_path: {}", spectrum_path);

    harm::HARMModel harm_model(photon_n, mass_unit);

    harm_model.read_file(harm_dump_path);

    harm_model.init();

    monty_rand::init(123);
    // monty_rand::init(std::chrono::system_clock::now().time_since_epoch().count());

    auto start = std::chrono::system_clock::now();
    auto start_0 = start;
    int n_super_photon_created = 0;
    int n_rate = 0;

    spdlog::info("Starting main loop");

    while (true) {
        auto [photon, quit] = harm_model.make_super_photon();
        if (quit) {
            break;
        }
        harm_model.track_super_photon(photon);

        ++n_super_photon_created;
        ++n_rate;

        std::chrono::duration<double> elapsed_seconds = std::chrono::system_clock::now() - start;
        if (elapsed_seconds.count() > 1.0) {
            double rate = n_rate / elapsed_seconds.count();
            auto [zone_x_1, zone_x_2] = harm_model.get_zone_x();
            spdlog::info("Rate {:.2f} ph/s, zone ({}, {})", rate, zone_x_1, zone_x_2);
            n_rate = 0;
            start = std::chrono::system_clock::now();
        }
    }
    auto stop = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = stop - start_0;
    spdlog::info("Final rate {:.2f} ph/s", n_super_photon_created / elapsed_seconds.count());
    spdlog::info("Super photons:");
    spdlog::info("\tcreated: {}", n_super_photon_created);
    spdlog::info("\tscattered: {}", harm_model.get_n_super_photon_scattered());
    spdlog::info("\trecorded: {}", harm_model.get_n_super_photon_recorded());

    harm_model.report_spectrum(n_super_photon_created, spectrum_path);

    return 0;
}
