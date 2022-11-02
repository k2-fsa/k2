/**
 * Copyright      2021  Xiaomi Corporation (authors: Fangjun Kuang)
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

#include <fstream>
#include <utility>
#include <vector>

#include "k2/csrc/log.h"
#include "k2/torch/csrc/wave_reader.h"

namespace k2 {
namespace {
// see http://soundfile.sapp.org/doc/WaveFormat/
//
// Note: We assume little endian here
// TODO(fangjun): Support big endian
struct WaveHeader {
  void Validate() const {
    //                       F F I R
    K2_CHECK_EQ(chunk_id, 0x46464952);
    //                     E V A W
    K2_CHECK_EQ(format, 0x45564157);
    K2_CHECK_EQ(subchunk1_id, 0x20746d66);
    K2_CHECK_EQ(subchunk1_size, 16);  // 16 for PCM
    K2_CHECK_EQ(audio_format, 1);     // 1 for PCM
    K2_CHECK_EQ(num_channels, 1);     // we support only single channel for now
    K2_CHECK_EQ(byte_rate, sample_rate * num_channels * bits_per_sample / 8)
        << "byte_rate: " << byte_rate << ", "
        << "sample_rate: " << sample_rate << ", "
        << "num_channels: " << num_channels << ", "
        << "bits_per_sample: " << bits_per_sample;
    K2_CHECK_EQ(block_align, num_channels * bits_per_sample / 8)
        << "block_align: " << block_align << ", "
        << "num_channels: " << num_channels << ", "
        << "bits_per_sample: " << bits_per_sample << ", ";
    K2_CHECK_EQ(bits_per_sample, 16);  // we support only 16 bits per sample
  }

  // See
  // https://en.wikipedia.org/wiki/WAV#Metadata
  // and
  // https://www.robotplanet.dk/audio/wav_meta_data/riff_mci.pdf
  void SeekToDataChunk(std::istream &is) {
    //                              a t a d
    while (is && subchunk2_id != 0x61746164) {
      const char *p = reinterpret_cast<const char *>(&subchunk2_id);
      is.seekg(subchunk2_size, std::istream::cur);
      is.read(reinterpret_cast<char *>(&subchunk2_id), sizeof(int32_t));
      is.read(reinterpret_cast<char *>(&subchunk2_size), sizeof(int32_t));
    }
  }

  int32_t chunk_id;
  int32_t chunk_size;
  int32_t format;
  int32_t subchunk1_id;
  int32_t subchunk1_size;
  int16_t audio_format;
  int16_t num_channels;
  int32_t sample_rate;
  int32_t byte_rate;
  int16_t block_align;
  int16_t bits_per_sample;
  int32_t subchunk2_id;
  int32_t subchunk2_size;
};
static_assert(sizeof(WaveHeader) == 44, "");

// Read a wave file of mono-channel.
// Return its samples in a 1-D torch.float32 tensor, normalized
// by dividing 32768.
// Also, it returns the sample rate.
std::pair<torch::Tensor, float> ReadWaveImpl(std::istream &is) {
  WaveHeader header;
  is.read(reinterpret_cast<char *>(&header), sizeof(header));
  K2_CHECK((bool)is) << "Failed to read wave header";

  header.Validate();

  header.SeekToDataChunk(is);
  K2_CHECK((bool)is) << "Failed to locate the data chunk";

  float sample_rate = header.sample_rate;

  // header.subchunk2_size contains the number of bytes in the data.
  // As we assume each sample contains two bytes, so it is divided by 2 here
  torch::Tensor data = torch::empty({header.subchunk2_size / 2}, torch::kShort);

  is.read(reinterpret_cast<char *>(data.data_ptr<int16_t>()),
          header.subchunk2_size);

  K2_CHECK((bool)is) << "Failed to read wave samples";
  data = (data / 32768.).to(torch::kFloat32);
  return {data, sample_rate};
}

}  // namespace

WaveReader::WaveReader(const std::string &filename) {
  std::ifstream is(filename, std::ifstream::binary);
  std::tie(data_, sample_rate_) = ReadWaveImpl(is);
}

WaveReader::WaveReader(std::istream &is) {
  std::tie(data_, sample_rate_) = ReadWaveImpl(is);
}

torch::Tensor ReadWave(const std::string &filename,
                       float expected_sample_rate) {
  WaveReader reader(filename);
  K2_CHECK_EQ(reader.SampleRate(), expected_sample_rate);
  return reader.Data();
}

std::vector<torch::Tensor> ReadWave(const std::vector<std::string> &filenames,
                                    float expected_sample_rate) {
  std::vector<torch::Tensor> ans;
  ans.reserve(filenames.size());
  for (const auto &path : filenames) {
    ans.emplace_back(ReadWave(path, expected_sample_rate));
  }
  return ans;
}

}  // namespace k2
