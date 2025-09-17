/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
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

    harm_model.run_simulation();

    harm_model.report_spectrum(spectrum_path);

    return 0;
}
