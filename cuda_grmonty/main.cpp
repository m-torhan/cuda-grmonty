/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iostream>
#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"

#include "cuda_grmonty/harm_data.hpp"

// Define cli args
ABSL_FLAG(int, photon_n, 5000000, "Estimate of photon number");
ABSL_FLAG(float, mass_unit, 4e19, "Mass unit");
ABSL_FLAG(std::string, harm_dump_path, "", "HARM dump file path");

int main(int argc, char *argv[]) {
    absl::ParseCommandLine(argc, argv);

    int photon_n = absl::GetFlag(FLAGS_photon_n);
    float mass_unit = absl::GetFlag(FLAGS_mass_unit);
    std::string harm_dump_path = absl::GetFlag(FLAGS_harm_dump_path);

    std::cout << "photon_n: " << photon_n << std::endl;
    std::cout << "mass_unit: " << mass_unit << std::endl;
    std::cout << "harm_dump_path: " << harm_dump_path << std::endl;

    harm::HARMData harm_data;
    harm_data.read_file(harm_dump_path);

    return 0;
}
