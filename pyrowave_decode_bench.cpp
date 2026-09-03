// Copyright (c) 2026 nibiru2010-sys (Pyrowave-vr fork)
// SPDX-License-Identifier: MIT

// Decode-time sweep for VR streaming targets (Quest 3 / Adreno 740).
//
// Per configuration (width x height x bpp x decode path): encode a synthetic
// worst-case frame once (mirror ramp, saturates rate control like the c-test
// signal), then decode it N times through the C API measuring wall-clock per
// decode plus GPU stage statistics. The wall clock includes packet upload and
// CPU readback of the YUV planes (upper bound); the GPU stage sum is the
// in-stream cost since the real client decodes into GPU images.
//
// Everything runs on-device, including the encode — it is setup-only, paid
// once per configuration, and exists just to produce a realistic bitstream.

#include "vulkan/vulkan.h"
#include "pyrowave.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <string>
#include <vector>

#define CHECKED(x) do { \
	pyrowave_result res_ = x; \
	if (res_ != PYROWAVE_SUCCESS) { fprintf(stderr, "Got pyrowave result %d while executing %s at line %d.\n", res_, #x, __LINE__); return false; } \
} while(false)

struct Config
{
	int width;
	int height;
	double bpp;
};

// Stereo frames (both eyes side by side), the shape ALVR would encode.
// Bitrates at 90 Hz: 6144x3208@1.0bpp = 1.77 Gbps, @1.5bpp = 2.66 Gbps
// (both under the measured 3.2 Gbps ADB tunnel ceiling).
static const Config default_matrix[] = {
	{ 6144, 3208, 0.75 },
	{ 6144, 3208, 1.0 },
	{ 6144, 3208, 1.5 },
	{ 5408, 2808, 1.0 },
	{ 4128, 2208, 1.0 },
	{ 3072, 1664, 1.0 },
};

static double now_ms()
{
	return std::chrono::duration<double, std::milli>(
			std::chrono::steady_clock::now().time_since_epoch()).count();
}

static uint8_t mirror(int v)
{
	v &= 511;
	if (v > 255)
		v = 511 - v;
	return uint8_t(v);
}

static void report_stats(pyrowave_device device, std::vector<std::string> *out)
{
	pyrowave_device_report_performance_stats(device, [](void *userdata, const char *msg)
	{
		static_cast<std::vector<std::string> *>(userdata)->push_back(msg);
	}, out, true);
}

static bool run_config(pyrowave_device device, const Config &cfg, bool fragment_path,
                       int warmup, int iters, const char *precision_label)
{
	pyrowave_encoder_create_info encoder_info = {};
	encoder_info.device = device;
	encoder_info.width = cfg.width;
	encoder_info.height = cfg.height;
	encoder_info.chroma = PYROWAVE_CHROMA_SUBSAMPLING_420;

	pyrowave_decoder_create_info decoder_info = {};
	decoder_info.device = device;
	decoder_info.width = cfg.width;
	decoder_info.height = cfg.height;
	decoder_info.chroma = PYROWAVE_CHROMA_SUBSAMPLING_420;
	decoder_info.fragment_path = fragment_path;

	pyrowave_encoder encoder;
	pyrowave_decoder decoder;
	CHECKED(pyrowave_encoder_create(&encoder_info, &encoder));
	CHECKED(pyrowave_decoder_create(&decoder_info, &decoder));

	const size_t luma_size = size_t(cfg.width) * cfg.height;
	const size_t chroma_size = luma_size / 4;
	const size_t max_bitstream = size_t(cfg.bpp / 8.0 * double(luma_size));

	// Synthetic worst-case NV12 input.
	std::vector<uint8_t> luma(luma_size);
	std::vector<uint16_t> cbcr(chroma_size);
	for (int y = 0; y < cfg.height; y++)
		for (int x = 0; x < cfg.width; x++)
			luma[size_t(y) * cfg.width + x] = mirror(3 * x + 5 * y);
	for (int y = 0; y < cfg.height / 2; y++)
		for (int x = 0; x < cfg.width / 2; x++)
			cbcr[size_t(y) * (cfg.width / 2) + x] =
					uint16_t(mirror(7 * x + 3 * y) | (mirror(3 * x + 7 * y) << 8));

	pyrowave_cpu_buffer encode_buffer = {};
	encode_buffer.format = PYROWAVE_CPU_BUFFER_FORMAT_NV12;
	encode_buffer.row_stride_in_bytes[0] = cfg.width;
	encode_buffer.row_stride_in_bytes[1] = cfg.width;
	encode_buffer.plane_size_in_bytes[0] = luma_size;
	encode_buffer.plane_size_in_bytes[1] = 2 * chroma_size;
	encode_buffer.data[0] = luma.data();
	encode_buffer.data[1] = cbcr.data();
	encode_buffer.width = cfg.width;
	encode_buffer.height = cfg.height;

	double encode_ms = now_ms();
	const pyrowave_rate_control rate_control = { max_bitstream };
	CHECKED(pyrowave_encoder_encode_cpu_synchronous(encoder, &encode_buffer, &rate_control));
	encode_ms = now_ms() - encode_ms;

	size_t num_packets;
	CHECKED(pyrowave_encoder_compute_num_packets(encoder, 8 * 1024, &num_packets));
	std::vector<uint8_t> bitstream(max_bitstream);
	std::vector<pyrowave_packet> packets(num_packets);
	CHECKED(pyrowave_encoder_packetize(encoder, packets.data(), 8 * 1024,
	                                   &num_packets, bitstream.data(), bitstream.size()));

	size_t actual_bytes = 0;
	for (auto &packet : packets)
		actual_bytes += packet.size;
	double effective_bpp = 8.0 * double(actual_bytes) / double(luma_size);

	// Drop the encode stage samples so decode stats are clean.
	std::vector<std::string> ignored;
	report_stats(device, &ignored);

	std::vector<uint8_t> decode_luma(luma_size);
	std::vector<uint8_t> decode_cb(chroma_size);
	std::vector<uint8_t> decode_cr(chroma_size);
	pyrowave_cpu_buffer decode_buffer = {};
	decode_buffer.format = PYROWAVE_CPU_BUFFER_FORMAT_YUV420P;
	decode_buffer.row_stride_in_bytes[0] = cfg.width;
	decode_buffer.row_stride_in_bytes[1] = cfg.width / 2;
	decode_buffer.row_stride_in_bytes[2] = cfg.width / 2;
	decode_buffer.plane_size_in_bytes[0] = luma_size;
	decode_buffer.plane_size_in_bytes[1] = chroma_size;
	decode_buffer.plane_size_in_bytes[2] = chroma_size;
	decode_buffer.data[0] = decode_luma.data();
	decode_buffer.data[1] = decode_cb.data();
	decode_buffer.data[2] = decode_cr.data();
	decode_buffer.width = cfg.width;
	decode_buffer.height = cfg.height;

	std::vector<double> samples;
	samples.reserve(iters);

	for (int iter = 0; iter < warmup + iters; iter++)
	{
		pyrowave_decoder_clear(decoder);
		for (auto &packet : packets)
			CHECKED(pyrowave_decoder_push_packet(decoder, bitstream.data() + packet.offset, packet.size));
		if (!pyrowave_decoder_decode_is_ready(decoder, false))
		{
			fprintf(stderr, "Decoder not ready after pushing all packets.\n");
			return false;
		}

		double start = now_ms();
		CHECKED(pyrowave_decoder_decode_cpu_buffer_synchronous(decoder, &decode_buffer));
		double elapsed = now_ms() - start;
		if (iter >= warmup)
			samples.push_back(elapsed);
	}

	std::sort(samples.begin(), samples.end());
	double avg = 0.0;
	for (double s : samples)
		avg += s;
	avg /= double(samples.size());
	double min = samples.front();
	double p95 = samples[size_t(0.95 * double(samples.size() - 1))];
	double max = samples.back();

	std::vector<std::string> gpu_stats;
	report_stats(device, &gpu_stats);

	printf("CONFIG res=%dx%d bpp=%.2f path=%s prec=%s bytes=%zu/%zu effbpp=%.2f packets=%zu encode_ms=%.1f\n",
	       cfg.width, cfg.height, cfg.bpp, fragment_path ? "fragment" : "compute", precision_label,
	       actual_bytes, max_bitstream, effective_bpp, num_packets, encode_ms);
	printf("WALL decode_ms avg=%.2f min=%.2f p95=%.2f max=%.2f fps_cap=%.1f\n",
	       avg, min, p95, max, 1000.0 / avg);
	for (auto &stat : gpu_stats)
		printf("GPU %s", stat.c_str());
	printf("CSV,%d,%d,%.2f,%s,%zu,%.2f,%.2f,%.2f,%.2f,%.1f\n",
	       cfg.width, cfg.height, cfg.bpp, fragment_path ? "fragment" : "compute",
	       actual_bytes, avg, min, p95, max, 1000.0 / avg);
	fflush(stdout);

	pyrowave_decoder_destroy(decoder);
	pyrowave_encoder_destroy(encoder);
	return true;
}

static void print_help()
{
	printf("Usage: pyrowave-decode-bench [--width W --height H --bpp B] | --matrix\n"
	       "\t[--path compute|fragment|both] (default: both)\n"
	       "\t[--iters N] (default 30) [--warmup N] (default 5)\n"
	       "\t[--label S] precision tag for the output header (PYROWAVE_PRECISION is\n"
	       "\tprivate to the library, so the binary cannot see its own build mode)\n"
	       "\tSingle-config mode needs all of width/height/bpp.\n");
}

int main(int argc, char **argv)
{
	int width = 0, height = 0;
	double bpp = 0.0;
	bool matrix = false;
	bool run_compute = true, run_fragment = true;
	int iters = 30, warmup = 5;
	const char *label = "unlabeled";

	for (int i = 1; i < argc; i++)
	{
		if (!strcmp(argv[i], "--width") && i + 1 < argc)
			width = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--height") && i + 1 < argc)
			height = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--bpp") && i + 1 < argc)
			bpp = atof(argv[++i]);
		else if (!strcmp(argv[i], "--path") && i + 1 < argc)
		{
			run_compute = run_fragment = false;
			if (!strcmp(argv[i + 1], "compute")) run_compute = true;
			else if (!strcmp(argv[i + 1], "fragment")) run_fragment = true;
			else { print_help(); return EXIT_FAILURE; }
			i++;
		}
		else if (!strcmp(argv[i], "--iters") && i + 1 < argc)
			iters = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--warmup") && i + 1 < argc)
			warmup = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--matrix"))
			matrix = true;
		else if (!strcmp(argv[i], "--label") && i + 1 < argc)
			label = argv[++i];
		else
		{
			print_help();
			return EXIT_FAILURE;
		}
	}

	std::vector<Config> configs;
	if (matrix || (width == 0 && height == 0 && bpp == 0.0))
		configs.assign(default_matrix, default_matrix + sizeof(default_matrix) / sizeof(default_matrix[0]));
	else if (width > 0 && height > 0 && bpp > 0.0)
		configs.push_back({ width, height, bpp });
	else
	{
		print_help();
		return EXIT_FAILURE;
	}

	uint32_t major, minor, patch;
	pyrowave_get_api_version(&major, &minor, &patch);

	pyrowave_device device;
	if (pyrowave_create_default_device(&device) != PYROWAVE_SUCCESS)
	{
		fprintf(stderr, "Failed to create Vulkan device.\n");
		return EXIT_FAILURE;
	}

	const char *precision_label = label;

	printf("# pyrowave-decode-bench api=%u.%u.%u precision=%s prefers_fragment=%d iters=%d warmup=%d\n",
	       major, minor, patch, precision_label,
	       pyrowave_decoder_device_prefers_fragment_path(device) ? 1 : 0, iters, warmup);
	fflush(stdout);

	int failures = 0;
	for (auto &cfg : configs)
	{
		if (run_compute && !run_config(device, cfg, false, warmup, iters, precision_label))
			failures++;
		if (run_fragment && !run_config(device, cfg, true, warmup, iters, precision_label))
			failures++;
	}

	pyrowave_device_destroy(device);
	printf("# done failures=%d\n", failures);
	return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
