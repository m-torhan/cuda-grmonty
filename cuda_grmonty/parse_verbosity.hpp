/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>

#include "absl/strings/string_view.h"
#include "spdlog/common.h"

namespace spdlog::level {

/**
 * Converts string to spdlog level enum
 *
 * @param text Logging verbosity in text format
 * @param level Logging verbosity in enum format
 * @param error Parsing error
 *
 * @returns True if verbosity was succesfull
 */
bool AbslParseFlag(absl::string_view text, level_enum *level, std::string *error);

/**
 * Converts spdlog level enum to string
 *
 * @param level Logging verbosity
 *
 * @returns Logging verbosity as string
 */
std::string AbslUnparseFlag(level_enum level);

}; /* namespace spdlog::level */
