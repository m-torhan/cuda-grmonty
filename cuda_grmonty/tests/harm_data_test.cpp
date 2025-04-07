/*
 * Copyright (c) 2025 Maciej Torhan <https://github.com/m-torhan>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cuda_grmonty/harm_data.hpp"
#include <format>
#include <fstream>
#include <gtest/gtest.h>
#include <string>

const harm::Header sample_header{
    .t = 1.0,
    .n = {2, 3},
    .x_start = {0.0, 4.0, 5.0, 0.0},
    .x_stop = {0.0, 20.0, 32.0, 0.0},
    .dx = {0.0, 8.0, 9.0, 0.0},
    .t_final = 10.0,
    .n_step = 11,
    .a = 12.0,
    .gamma = 13.0,
    .courant = 14.0,
    .dt_dump = 15.0,
    .dt_log = 16.0,
    .dt_img = 17.0,
    .dt_rdump = 18,
    .cnt_dump = 19,
    .cnt_img = 20,
    .cnt_rdump = 21,
    .dt = 22.0,
    .lim = 23,
    .failed = 24,
    .r_in = 25.0,
    .r_out = 26.0,
    .h_slope = 27.0,
    .r_0 = 28.0,
};

const harm::Data sample_data{
    .p = {{111.0, 112.0, 113.0}, {121.0, 122.0, 123.0}},
    .u = {{211.0, 212.0, 213.0}, {221.0, 222.0, 223.0}},
    .u_1 = {{311.0, 312.0, 313.0}, {321.0, 322.0, 323.0}},
    .u_2 = {{411.0, 412.0, 413.0}, {421.0, 422.0, 423.0}},
    .u_3 = {{511.0, 512.0, 513.0}, {521.0, 522.0, 523.0}},
    .b_1 = {{611.0, 612.0, 613.0}, {621.0, 622.0, 623.0}},
    .b_2 = {{711.0, 712.0, 713.0}, {721.0, 722.0, 723.0}},
    .b_3 = {{811.0, 812.0, 813.0}, {821.0, 822.0, 823.0}},
};

/**
 * Writes sample HARM header to file
 *
 * @param file Output file
 */
static void write_sample_harm_header(std::ofstream &file);

/**
 * Writes sample HARM data to file
 *
 * @param file Output file
 */
static void write_sample_harm_data(std::ofstream &file);

/**
 * Test reading header from file
 */
TEST(HARMData, ReadFileHeader) {
    std::string filepath = testing::TempDir() + "harm_dump";

    /* write data to file */
    std::ofstream file(filepath);

    ASSERT_TRUE(file.is_open());

    write_sample_harm_header(file);

    file.close();

    /* read file */
    harm::HARMData harm_data;

    harm_data.read_file(filepath);

    /* check values */
    ASSERT_DOUBLE_EQ(sample_header.t, harm_data.get_header()->t);
    ASSERT_EQ(sample_header.n[0], harm_data.get_header()->n[0]);
    ASSERT_EQ(sample_header.n[1], harm_data.get_header()->n[1]);
    ASSERT_DOUBLE_EQ(sample_header.x_start[1], harm_data.get_header()->x_start[1]);
    ASSERT_DOUBLE_EQ(sample_header.x_start[2], harm_data.get_header()->x_start[2]);
    ASSERT_DOUBLE_EQ(sample_header.x_stop[1], harm_data.get_header()->x_stop[1]);
    ASSERT_DOUBLE_EQ(sample_header.x_stop[2], harm_data.get_header()->x_stop[2]);
    ASSERT_DOUBLE_EQ(sample_header.dx[1], harm_data.get_header()->dx[1]);
    ASSERT_DOUBLE_EQ(sample_header.dx[2], harm_data.get_header()->dx[2]);
    ASSERT_DOUBLE_EQ(sample_header.t_final, harm_data.get_header()->t_final);
    ASSERT_EQ(sample_header.n_step, harm_data.get_header()->n_step);
    ASSERT_DOUBLE_EQ(sample_header.a, harm_data.get_header()->a);
    ASSERT_DOUBLE_EQ(sample_header.gamma, harm_data.get_header()->gamma);
    ASSERT_DOUBLE_EQ(sample_header.courant, harm_data.get_header()->courant);
    ASSERT_DOUBLE_EQ(sample_header.dt_dump, harm_data.get_header()->dt_dump);
    ASSERT_DOUBLE_EQ(sample_header.dt_log, harm_data.get_header()->dt_log);
    ASSERT_DOUBLE_EQ(sample_header.dt_img, harm_data.get_header()->dt_img);
    ASSERT_EQ(sample_header.dt_rdump, harm_data.get_header()->dt_rdump);
    ASSERT_EQ(sample_header.cnt_dump, harm_data.get_header()->cnt_dump);
    ASSERT_EQ(sample_header.cnt_img, harm_data.get_header()->cnt_img);
    ASSERT_EQ(sample_header.cnt_rdump, harm_data.get_header()->cnt_rdump);
    ASSERT_DOUBLE_EQ(sample_header.dt, harm_data.get_header()->dt);
    ASSERT_EQ(sample_header.lim, harm_data.get_header()->lim);
    ASSERT_EQ(sample_header.failed, harm_data.get_header()->failed);
    ASSERT_DOUBLE_EQ(sample_header.r_in, harm_data.get_header()->r_in);
    ASSERT_DOUBLE_EQ(sample_header.r_out, harm_data.get_header()->r_out);
    ASSERT_DOUBLE_EQ(sample_header.h_slope, harm_data.get_header()->h_slope);
    ASSERT_DOUBLE_EQ(sample_header.r_0, harm_data.get_header()->r_0);
}

/**
 * Test reading data from file
 */
TEST(HARMData, ReadFileData) {
    std::string filepath = testing::TempDir() + "harm_dump";

    /* write data to file */
    std::ofstream file(filepath);

    ASSERT_TRUE(file.is_open());

    write_sample_harm_header(file);
    write_sample_harm_data(file);

    file.close();

    /* read file */
    harm::HARMData harm_data;

    system((std::string("cat ") + filepath).c_str());

    harm_data.read_file(filepath);

    /* check values */
    ASSERT_EQ(sample_header.n[0], harm_data.get_data()->p.size());
    ASSERT_EQ(sample_header.n[0], harm_data.get_data()->u.size());
    ASSERT_EQ(sample_header.n[0], harm_data.get_data()->u_1.size());
    ASSERT_EQ(sample_header.n[0], harm_data.get_data()->u_2.size());
    ASSERT_EQ(sample_header.n[0], harm_data.get_data()->u_3.size());
    ASSERT_EQ(sample_header.n[0], harm_data.get_data()->b_1.size());
    ASSERT_EQ(sample_header.n[0], harm_data.get_data()->b_2.size());
    ASSERT_EQ(sample_header.n[0], harm_data.get_data()->b_3.size());

    for (int i = 0; i < sample_header.n[0]; ++i) {
        ASSERT_EQ(sample_header.n[1], harm_data.get_data()->p[i].size());
        ASSERT_EQ(sample_header.n[1], harm_data.get_data()->u[i].size());
        ASSERT_EQ(sample_header.n[1], harm_data.get_data()->u_1[i].size());
        ASSERT_EQ(sample_header.n[1], harm_data.get_data()->u_2[i].size());
        ASSERT_EQ(sample_header.n[1], harm_data.get_data()->u_3[i].size());
        ASSERT_EQ(sample_header.n[1], harm_data.get_data()->b_1[i].size());
        ASSERT_EQ(sample_header.n[1], harm_data.get_data()->b_2[i].size());
        ASSERT_EQ(sample_header.n[1], harm_data.get_data()->b_3[i].size());
    }

    for (int i = 0; i < sample_header.n[0]; ++i) {
        for (int j = 0; j < sample_header.n[1]; ++j) {
            ASSERT_DOUBLE_EQ(sample_data.p[i][j], harm_data.get_data()->p[i][j]);
            ASSERT_DOUBLE_EQ(sample_data.u[i][j], harm_data.get_data()->u[i][j]);
            ASSERT_DOUBLE_EQ(sample_data.u_1[i][j], harm_data.get_data()->u_1[i][j]);
            ASSERT_DOUBLE_EQ(sample_data.u_2[i][j], harm_data.get_data()->u_2[i][j]);
            ASSERT_DOUBLE_EQ(sample_data.u_2[i][j], harm_data.get_data()->u_2[i][j]);
            ASSERT_DOUBLE_EQ(sample_data.b_1[i][j], harm_data.get_data()->b_1[i][j]);
            ASSERT_DOUBLE_EQ(sample_data.b_2[i][j], harm_data.get_data()->b_2[i][j]);
            ASSERT_DOUBLE_EQ(sample_data.b_2[i][j], harm_data.get_data()->b_2[i][j]);
        }
    }
}

static void write_sample_harm_header(std::ofstream &file) {
    file << std::format("{} ", sample_header.t);
    file << std::format("{} {} ", sample_header.n[0], sample_header.n[1]);
    file << std::format("{} {} ", sample_header.x_start[1], sample_header.x_start[2]);
    file << std::format("{} {} ", sample_header.dx[1], sample_header.dx[2]);
    file << std::format("{} {} ", sample_header.t_final, sample_header.n_step);
    file << std::format("{} {} {} ", sample_header.a, sample_header.gamma, sample_header.courant);
    file << std::format("{} {} {} ", sample_header.dt_dump, sample_header.dt_log, sample_header.dt_img);
    file << std::format("{} ", sample_header.dt_rdump);
    file << std::format("{} {} {} ", sample_header.cnt_dump, sample_header.cnt_img, sample_header.cnt_rdump);
    file << std::format("{} {} {} ", sample_header.dt, sample_header.lim, sample_header.failed);
    file << std::format("{} {} ", sample_header.r_in, sample_header.r_out);
    file << std::format("{} {}", sample_header.h_slope, sample_header.r_0);
    file << std::endl;
    file.flush();
}

static void write_sample_harm_data(std::ofstream &file) {
    for (int j = 0; j < sample_header.n[1]; ++j) {
        for (int i = 0; i < sample_header.n[0]; ++i) {
            file << "0 0 0 0 "; /* x[1], x[2], r, h */
            file << std::format("{} ", sample_data.p[i][j]);
            file << std::format("{} ", sample_data.u[i][j]);
            file << std::format("{} ", sample_data.u_1[i][j]);
            file << std::format("{} ", sample_data.u_2[i][j]);
            file << std::format("{} ", sample_data.u_3[i][j]);
            file << std::format("{} ", sample_data.b_1[i][j]);
            file << std::format("{} ", sample_data.b_2[i][j]);
            file << std::format("{} ", sample_data.b_3[i][j]);
            file << "0 ";               /* div_b */
            file << "0 0 0 0 0 0 0 0 "; /* u_con, u_cov */
            file << "0 0 0 0 0 0 0 0 "; /* b_con, b_cov */
            file << "0 0 0 0 ";         /* vmin, vmax */
            file << "0";                /* g_det */
            file << std::endl;
        }
    }
    file.flush();
}
