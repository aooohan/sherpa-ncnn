/**
 * Copyright (c)  2024  Xiaomi Corporation (authors: Fangjun Kuang)
 *
 * See LICENSE for clarification regarding multiple authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// This file demonstrates how to use VAD with sherpa-ncnn's C API.
// clang-format off
//
// Usage:
//  ./bin/vad-c-api \
//    /path/to/silero-vad-model-dir \
//    /path/to/foo.wav
//
// To download the VAD model:
//  wget https://github.com/k2-fsa/sherpa-ncnn/releases/download/models/silero-vad-ncnn.tar.bz2
//  tar xvf silero-vad-ncnn.tar.bz2
//
// clang-format on

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sherpa-ncnn/c-api/c-api.h"

const char *kUsage =
    "\n"
    "Usage:\n"
    "  ./bin/vad-c-api \\\n"
    "    /path/to/silero-vad-model-dir \\\n"
    "    /path/to/foo.wav\n"
    "\n"
    "The model directory should contain silero.ncnn.param and "
    "silero.ncnn.bin.\n"
    "\n"
    "Please refer to\n"
    "https://k2-fsa.github.io/sherpa/ncnn/\n"
    "for more information.\n";

int32_t main(int32_t argc, char *argv[]) {
  if (argc != 3) {
    fprintf(stderr, "%s\n", kUsage);
    return -1;
  }

  const char *model_dir = argv[1];
  const char *wav_filename = argv[2];

  FILE *fp = fopen(wav_filename, "rb");
  if (!fp) {
    fprintf(stderr, "Failed to open %s\n", wav_filename);
    return -1;
  }

  // Assume the wave header occupies 44 bytes.
  fseek(fp, 44, SEEK_SET);

  // Get file size
  fseek(fp, 0, SEEK_END);
  long file_size = ftell(fp);
  fseek(fp, 44, SEEK_SET);

  int32_t num_samples = (file_size - 44) / sizeof(int16_t);
  int16_t *buffer = (int16_t *)malloc(num_samples * sizeof(int16_t));
  float *samples = (float *)malloc(num_samples * sizeof(float));

  size_t n = fread((void *)buffer, sizeof(int16_t), num_samples, fp);
  fclose(fp);

  if (n != (size_t)num_samples) {
    fprintf(stderr, "Failed to read %s\n", wav_filename);
    free(buffer);
    free(samples);
    return -1;
  }

  // Convert int16 to float and normalize to [-1, 1]
  for (int32_t i = 0; i < num_samples; ++i) {
    samples[i] = buffer[i] / 32768.0f;
  }
  free(buffer);

  // Configure VAD
  SherpaNcnnVadModelConfig vad_config;
  memset(&vad_config, 0, sizeof(vad_config));

  vad_config.model_dir = model_dir;
  vad_config.threshold = 0.5f;
  vad_config.min_silence_duration = 0.5f;
  vad_config.min_speech_duration = 0.25f;
  vad_config.window_size = 512;
  vad_config.sample_rate = 16000;
  vad_config.use_vulkan_compute = 0;
  vad_config.num_threads = 1;

  float buffer_size_in_seconds = 60.0f;

  SherpaNcnnVoiceActivityDetector *vad =
      SherpaNcnnCreateVoiceActivityDetector(&vad_config, buffer_size_in_seconds);

  if (vad == NULL) {
    fprintf(stderr, "Failed to create VAD. Please check your config.\n");
    free(samples);
    return -1;
  }

  fprintf(stderr, "Started VAD processing...\n");

  int32_t window_size = vad_config.window_size;
  int32_t i = 0;
  int32_t is_eof = 0;
  int32_t segment_index = 0;

  while (!is_eof) {
    if (i + window_size < num_samples) {
      SherpaNcnnVoiceActivityDetectorAcceptWaveform(vad, samples + i,
                                                    window_size);
    } else {
      // Process remaining samples and flush
      if (i < num_samples) {
        SherpaNcnnVoiceActivityDetectorAcceptWaveform(vad, samples + i,
                                                      num_samples - i);
      }
      SherpaNcnnVoiceActivityDetectorFlush(vad);
      is_eof = 1;
    }

    // Process detected speech segments
    while (!SherpaNcnnVoiceActivityDetectorEmpty(vad)) {
      const SherpaNcnnSpeechSegment *segment =
          SherpaNcnnVoiceActivityDetectorFront(vad);

      float start = segment->start / 16000.0f;
      float duration = segment->n / 16000.0f;
      float stop = start + duration;

      fprintf(stderr, "Segment %d: %.3f -- %.3f (duration: %.3f seconds)\n",
              segment_index, start, stop, duration);

      segment_index++;

      SherpaNcnnDestroySpeechSegment(segment);
      SherpaNcnnVoiceActivityDetectorPop(vad);
    }

    i += window_size;
  }

  fprintf(stderr, "\nTotal speech segments detected: %d\n", segment_index);

  SherpaNcnnDestroyVoiceActivityDetector(vad);
  free(samples);

  return 0;
}
