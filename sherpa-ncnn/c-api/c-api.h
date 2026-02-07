/**
 * Copyright (c)  2023  Xiaomi Corporation (authors: Fangjun Kuang)
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

#ifndef SHERPA_NCNN_C_API_C_API_H_
#define SHERPA_NCNN_C_API_C_API_H_

// C API for sherpa-ncnn
//
// Please refer to
// https://github.com/k2-fsa/sherpa-ncnn/blob/master/c-api-examples/decode-file-c-api.c
// for usages.

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// See https://github.com/pytorch/pytorch/blob/main/c10/macros/Export.h
// We will set SHERPA_NCNN_BUILD_SHARED_LIBS and SHERPA_NCNN_BUILD_MAIN_LIB in
// CMakeLists.txt

#if defined(_WIN32)
#if defined(SHERPA_NCNN_BUILD_SHARED_LIBS)
#define SHERPA_NCNN_EXPORT __declspec(dllexport)
#define SHERPA_NCNN_IMPORT __declspec(dllimport)
#else
#define SHERPA_NCNN_EXPORT
#define SHERPA_NCNN_IMPORT
#endif
#else  // WIN32
#define SHERPA_NCNN_EXPORT __attribute__((visibility("default")))
#define SHERPA_NCNN_IMPORT SHERPA_NCNN_EXPORT
#endif

#if defined(SHERPA_NCNN_BUILD_MAIN_LIB)
#define SHERPA_NCNN_API SHERPA_NCNN_EXPORT
#else
#define SHERPA_NCNN_API SHERPA_NCNN_IMPORT
#endif

// Please don't free the returned pointer.
// Please don't modify the memory pointed by the returned pointer.
//
// The memory pointed by the returned pointer is statically allocated.
//
// Example return value: "1.12.1"
SHERPA_NCNN_API const char *SherpaNcnnGetVersionStr();

// Please don't free the returned pointer.
// Please don't modify the memory pointed by the returned pointer.
//
// The memory pointed by the returned pointer is statically allocated.
//
// Example return value: "6982b86c"
SHERPA_NCNN_API const char *SherpaNcnnGetGitSha1();

// Please don't free the returned pointer.
// Please don't modify the memory pointed by the returned pointer.
//
// The memory pointed by the returned pointer is statically allocated.
//
// Example return value: "Fri Jun 20 11:22:52 2025"
SHERPA_NCNN_API const char *SherpaNcnnGetGitDate();

/// Please refer to
/// https://k2-fsa.github.io/sherpa/ncnn/pretrained_models/index.html
/// to download pre-trained models. That is, you can find .ncnn.param,
/// .ncnn.bin, and tokens.txt for this struct from there.
SHERPA_NCNN_API typedef struct SherpaNcnnModelConfig {
  /// Path to encoder.ncnn.param
  const char *encoder_param;

  /// Path to encoder.ncnn.bin
  const char *encoder_bin;

  /// Path to decoder.ncnn.param
  const char *decoder_param;

  /// Path to decoder.ncnn.bin
  const char *decoder_bin;

  /// Path to joiner.ncnn.param
  const char *joiner_param;

  /// Path to joiner.ncnn.bin
  const char *joiner_bin;

  /// Path to tokens.txt
  const char *tokens;

  /// If it is non-zero, and it has GPU available, and ncnn is built
  /// with vulkan, then it will use GPU for computation.
  /// Otherwise, it uses CPU for computation.
  int32_t use_vulkan_compute;

  /// Number of threads for neural network computation.
  int32_t num_threads;
} SherpaNcnnModelConfig;

SHERPA_NCNN_API typedef struct SherpaNcnnDecoderConfig {
  /// Decoding method. Supported values are:
  /// greedy_search, modified_beam_search
  const char *decoding_method;

  /// Number of active paths for modified_beam_search.
  /// It is ignored when decoding_method is greedy_search.
  int32_t num_active_paths;
} SherpaNcnnDecoderConfig;

SHERPA_NCNN_API typedef struct SherpaNcnnFeatureExtractorConfig {
  // Sampling rate of the input audio samples. MUST match the one
  // expected by the model. For instance, it should be 16000 for models
  // from icefall.
  float sampling_rate;

  // feature dimension. Must match the one expected by the model.
  // For instance, it should be 80 for models from icefall.
  int32_t feature_dim;
} SherpaNcnnFeatureExtractorConfig;

SHERPA_NCNN_API typedef struct SherpaNcnnRecognizerConfig {
  SherpaNcnnFeatureExtractorConfig feat_config;
  SherpaNcnnModelConfig model_config;
  SherpaNcnnDecoderConfig decoder_config;

  /// 0 to disable endpoint detection.
  /// A non-zero value to enable endpoint detection.
  int32_t enable_endpoint;

  /// An endpoint is detected if trailing silence in seconds is larger than
  /// this value even if nothing has been decoded.
  /// Used only when enable_endpoint is not 0.
  float rule1_min_trailing_silence;

  /// An endpoint is detected if trailing silence in seconds is larger than
  /// this value after something that is not blank has been decoded.
  /// Used only when enable_endpoint is not 0.
  float rule2_min_trailing_silence;

  /// An endpoint is detected if the utterance in seconds is larger than
  /// this value.
  /// Used only when enable_endpoint is not 0.
  float rule3_min_utterance_length;

  /// hotwords file, each line is a hotword which is segmented into char by
  /// space if language is something like CJK, segment manually, if language is
  /// something like English, segment by bpe model.
  const char *hotwords_file;

  /// scale of hotwords, used only when hotwords_file is not empty
  float hotwords_score;
} SherpaNcnnRecognizerConfig;

SHERPA_NCNN_API typedef struct SherpaNcnnResult {
  // Recognized text
  const char *text;

  // Pointer to continuous memory which holds string based tokens
  // which are seperated by \0
  const char *tokens;

  // Pointer to continuous memory which holds timestamps which
  // are seperated by \0
  float *timestamps;

  // The number of tokens/timestamps in above pointer
  int32_t count;
} SherpaNcnnResult;

SHERPA_NCNN_API typedef struct SherpaNcnnRecognizer SherpaNcnnRecognizer;
SHERPA_NCNN_API typedef struct SherpaNcnnStream SherpaNcnnStream;

/// Create a recognizer.
///
/// @param config  Config for the recognizer.
/// @return Return a pointer to the recognizer. The user has to invoke
//          DestroyRecognizer() to free it to avoid memory leak.
SHERPA_NCNN_API SherpaNcnnRecognizer *CreateRecognizer(
    const SherpaNcnnRecognizerConfig *config);

/// Free a pointer returned by CreateRecognizer()
///
/// @param p A pointer returned by CreateRecognizer()
SHERPA_NCNN_API void DestroyRecognizer(SherpaNcnnRecognizer *p);

/// Create a stream for accepting audio samples
///
/// @param p A pointer returned by CreateRecognizer
/// @return Return a pointer to a stream. The caller MUST invoke
///         DestroyStream at the end to avoid memory leak.
SHERPA_NCNN_API SherpaNcnnStream *CreateStream(SherpaNcnnRecognizer *p);

SHERPA_NCNN_API void DestroyStream(SherpaNcnnStream *s);

/// Accept input audio samples and compute the features.
///
/// @param s  A pointer returned by CreateStream().
/// @param sample_rate  Sample rate of the input samples. If it is different
///                     from feat_config.sampling_rate, we will do resample.
///                     Caution: You MUST not use a different sampling_rate
///                     across different calls to AcceptWaveform()
/// @param samples A pointer to a 1-D array containing audio samples.
///                The range of samples has to be normalized to [-1, 1].
/// @param n  Number of elements in the samples array.
SHERPA_NCNN_API void AcceptWaveform(SherpaNcnnStream *s, float sample_rate,
                                    const float *samples, int32_t n);

/// Test if the stream has enough frames for decoding.
///
/// The common usage is:
///   while (IsReady(p, s)) {
///      Decode(p, s);
///   }
/// @param p A pointer returned by CreateRecognizer()
/// @param s A pointer returned by CreateStream()
/// @return Return 1 if the given stream is ready for decoding.
///         Return 0 otherwise.
SHERPA_NCNN_API int32_t IsReady(SherpaNcnnRecognizer *p, SherpaNcnnStream *s);

/// Pre-condition for this function:
///   You must ensure that IsReady(p, s) return 1 before calling this function.
///
/// @param p A pointer returned by CreateRecognizer()
/// @param s A pointer returned by CreateStream()
SHERPA_NCNN_API void Decode(SherpaNcnnRecognizer *p, SherpaNcnnStream *s);

/// Get the decoding results so far.
///
/// @param p A pointer returned by CreateRecognizer().
/// @param s A pointer returned by CreateStream()
/// @return A pointer containing the result. The user has to invoke
///         DestroyResult() to free the returned pointer to avoid memory leak.
SHERPA_NCNN_API SherpaNcnnResult *GetResult(SherpaNcnnRecognizer *p,
                                            SherpaNcnnStream *s);

/// Destroy the pointer returned by GetResult().
///
/// @param r A pointer returned by GetResult()
SHERPA_NCNN_API void DestroyResult(const SherpaNcnnResult *r);

/// Reset a stream
///
/// @param p A pointer returned by CreateRecognizer().
/// @param s A pointer returned by CreateStream().
SHERPA_NCNN_API void Reset(SherpaNcnnRecognizer *p, SherpaNcnnStream *s);

/// Signal that no more audio samples would be available.
/// After this call, you cannot call AcceptWaveform() any more.
///
/// @param s A pointer returned by CreateStream()
SHERPA_NCNN_API void InputFinished(SherpaNcnnStream *s);

/// Return 1 is an endpoint has been detected.
///
/// Common usage:
///   if (IsEndpoint(p, s)) {
///     Reset(p, s);
///   }
///
/// @param p A pointer returned by CreateRecognizer()
/// @return Return 1 if an endpoint is detected. Return 0 otherwise.
SHERPA_NCNN_API int32_t IsEndpoint(SherpaNcnnRecognizer *p,
                                   SherpaNcnnStream *s);

// for displaying results on Linux/macOS.
SHERPA_NCNN_API typedef struct SherpaNcnnDisplay SherpaNcnnDisplay;

// ============================================================
// For Voice Activity Detection (VAD)
// ============================================================

/// Configuration for Silero VAD model
SHERPA_NCNN_API typedef struct SherpaNcnnVadModelConfig {
  /// Path to the directory containing silero.ncnn.param and silero.ncnn.bin
  const char *model_dir;

  /// Threshold to classify a segment as speech.
  /// If the predicted probability of a segment is larger than this value,
  /// then it is classified as speech.
  /// Default: 0.5
  float threshold;

  /// Minimum silence duration in seconds.
  /// If the duration of silence is less than this value, the silence segment
  /// is not considered as a boundary.
  /// Default: 0.5
  float min_silence_duration;

  /// Minimum speech duration in seconds.
  /// If the duration of speech is less than this value, it is considered
  /// as noise and discarded.
  /// Default: 0.25
  float min_speech_duration;

  /// 512, 1024, 1536 samples for 16000 Hz
  /// 256, 512, 768 samples for 8000 Hz
  /// Default: 512
  int32_t window_size;

  /// Sample rate of the input audio. Can be 8000 or 16000.
  /// Default: 16000
  int32_t sample_rate;

  /// If it is non-zero, and GPU is available, and ncnn is built with Vulkan,
  /// then it will use GPU for computation. Otherwise, it uses CPU.
  int32_t use_vulkan_compute;

  /// Number of threads for neural network computation.
  /// Default: 1
  int32_t num_threads;
} SherpaNcnnVadModelConfig;

/// Represents a speech segment detected by VAD.
SHERPA_NCNN_API typedef struct SherpaNcnnSpeechSegment {
  /// The start sample index of this segment in the original audio.
  int32_t start;

  /// Pointer to the audio samples of this segment.
  /// The samples are normalized to [-1, 1].
  const float *samples;

  /// Number of samples in this segment.
  int32_t n;
} SherpaNcnnSpeechSegment;

/// Opaque pointer for the Voice Activity Detector.
SHERPA_NCNN_API typedef struct SherpaNcnnVoiceActivityDetector
    SherpaNcnnVoiceActivityDetector;

/// Create a voice activity detector.
///
/// @param config Configuration for the VAD.
/// @param buffer_size_in_seconds Size of the internal circular buffer in
///                               seconds. Default: 60.
/// @return Return a pointer to the VAD. The user has to invoke
///         SherpaNcnnDestroyVoiceActivityDetector() to free it to avoid
///         memory leak.
SHERPA_NCNN_API SherpaNcnnVoiceActivityDetector *
SherpaNcnnCreateVoiceActivityDetector(const SherpaNcnnVadModelConfig *config,
                                      float buffer_size_in_seconds);

/// Free a pointer returned by SherpaNcnnCreateVoiceActivityDetector().
///
/// @param p A pointer returned by SherpaNcnnCreateVoiceActivityDetector().
SHERPA_NCNN_API void SherpaNcnnDestroyVoiceActivityDetector(
    SherpaNcnnVoiceActivityDetector *p);

/// Accept input audio samples.
///
/// @param p A pointer returned by SherpaNcnnCreateVoiceActivityDetector().
/// @param samples A pointer to a 1-D array containing audio samples.
///                The range of samples has to be normalized to [-1, 1].
/// @param n Number of elements in the samples array.
SHERPA_NCNN_API void SherpaNcnnVoiceActivityDetectorAcceptWaveform(
    SherpaNcnnVoiceActivityDetector *p, const float *samples, int32_t n);

/// Check whether the speech segment queue is empty.
///
/// @param p A pointer returned by SherpaNcnnCreateVoiceActivityDetector().
/// @return Return 1 if the queue is empty; return 0 otherwise.
SHERPA_NCNN_API int32_t
SherpaNcnnVoiceActivityDetectorEmpty(SherpaNcnnVoiceActivityDetector *p);

/// Return the first speech segment in the queue.
/// It throws if the queue is empty.
///
/// @param p A pointer returned by SherpaNcnnCreateVoiceActivityDetector().
/// @return Return the first speech segment. The user has to invoke
///         SherpaNcnnDestroySpeechSegment() to free the returned pointer to
///         avoid memory leak.
SHERPA_NCNN_API const SherpaNcnnSpeechSegment *
SherpaNcnnVoiceActivityDetectorFront(SherpaNcnnVoiceActivityDetector *p);

/// Remove the first speech segment from the queue.
/// It throws if the queue is empty.
///
/// @param p A pointer returned by SherpaNcnnCreateVoiceActivityDetector().
SHERPA_NCNN_API void SherpaNcnnVoiceActivityDetectorPop(
    SherpaNcnnVoiceActivityDetector *p);

/// Clear the internal queue, removing all speech segments.
///
/// @param p A pointer returned by SherpaNcnnCreateVoiceActivityDetector().
SHERPA_NCNN_API void SherpaNcnnVoiceActivityDetectorClear(
    SherpaNcnnVoiceActivityDetector *p);

/// Reset the voice activity detector.
///
/// @param p A pointer returned by SherpaNcnnCreateVoiceActivityDetector().
SHERPA_NCNN_API void SherpaNcnnVoiceActivityDetectorReset(
    SherpaNcnnVoiceActivityDetector *p);

/// At the end of the utterance, invoke this method so that the last
/// speech segment can be detected.
///
/// @param p A pointer returned by SherpaNcnnCreateVoiceActivityDetector().
SHERPA_NCNN_API void SherpaNcnnVoiceActivityDetectorFlush(
    SherpaNcnnVoiceActivityDetector *p);

/// Check whether speech is currently detected.
///
/// @param p A pointer returned by SherpaNcnnCreateVoiceActivityDetector().
/// @return Return 1 if speech is detected; return 0 otherwise.
SHERPA_NCNN_API int32_t
SherpaNcnnVoiceActivityDetectorDetected(SherpaNcnnVoiceActivityDetector *p);

/// Free the pointer returned by SherpaNcnnVoiceActivityDetectorFront().
///
/// @param p A pointer returned by SherpaNcnnVoiceActivityDetectorFront().
SHERPA_NCNN_API void SherpaNcnnDestroySpeechSegment(
    const SherpaNcnnSpeechSegment *p);

/// Create a display object. Must be freed using DestroyDisplay to avoid
/// memory leak.
SHERPA_NCNN_API SherpaNcnnDisplay *CreateDisplay(int32_t max_word_per_line);

SHERPA_NCNN_API void DestroyDisplay(SherpaNcnnDisplay *display);

/// Print the result.
SHERPA_NCNN_API void SherpaNcnnPrint(SherpaNcnnDisplay *display, int32_t idx,
                                     const char *s);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // SHERPA_NCNN_C_API_C_API_H_
