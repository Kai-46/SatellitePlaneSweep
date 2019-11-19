// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#include "ply.h"
#include "psl_base/exception.h"

#include <fstream>
#include <algorithm>

namespace PSL {

template<typename T>
T ReverseBytes(const T &data) {
  T data_reversed = data;
  std::reverse(reinterpret_cast<char *>(&data_reversed),
               reinterpret_cast<char *>(&data_reversed) + sizeof(T));
  return data_reversed;
}

inline bool IsLittleEndian() {
#ifdef BOOST_BIG_ENDIAN
  return false;
#else
  return true;
#endif
}

template<typename T>
T NativeToLittleEndian(const T x) {
  if (IsLittleEndian()) {
    return x;
  } else {
    return ReverseBytes(x);
  }
}

template<typename T>
void WriteBinaryLittleEndian(std::ostream *stream, const T &data) {
  const T data_little_endian = NativeToLittleEndian(data);
  stream->write(reinterpret_cast<const char *>(&data_little_endian), sizeof(T));
}

void WriteTextPlyPoints(const std::string &path, const std::vector<PlyPoint> &points) {
  std::ofstream file(path);
  if (!file.is_open()) {
    PSL_THROW_EXCEPTION("Failed to open file")
  }

  file << "ply" << std::endl;
  file << "format ascii 1.0" << std::endl;
  file << "element vertex " << points.size() << std::endl;

  file << "property float x" << std::endl;
  file << "property float y" << std::endl;
  file << "property float z" << std::endl;

  file << "property uchar red" << std::endl;
  file << "property uchar green" << std::endl;
  file << "property uchar blue" << std::endl;

  file << "end_header" << std::endl;

  for (const auto &point : points) {
    file << point.x << " " << point.y << " " << point.z;

    file << " " << static_cast<int>(point.r) << " "
         << static_cast<int>(point.g) << " " << static_cast<int>(point.b);

    file << std::endl;
  }

  file.close();
}

void WriteBinaryPlyPoints(const std::string &path,
                          const std::vector<PlyPoint> &points) {
  std::fstream text_file(path, std::ios::out);
  if (!text_file.is_open()) {
    PSL_THROW_EXCEPTION("Failed to open file")
  }

  text_file << "ply" << std::endl;
  text_file << "format binary_little_endian 1.0" << std::endl;
  text_file << "element vertex " << points.size() << std::endl;

  text_file << "property float x" << std::endl;
  text_file << "property float y" << std::endl;
  text_file << "property float z" << std::endl;

  text_file << "property uchar red" << std::endl;
  text_file << "property uchar green" << std::endl;
  text_file << "property uchar blue" << std::endl;

  text_file << "end_header" << std::endl;
  text_file.close();

  std::fstream binary_file(path,
                           std::ios::out | std::ios::binary | std::ios::app);
  if (!binary_file.is_open()) {
    PSL_THROW_EXCEPTION("Failed to open file")
  }

  for (const auto &point : points) {
    WriteBinaryLittleEndian<float>(&binary_file, point.x);
    WriteBinaryLittleEndian<float>(&binary_file, point.y);
    WriteBinaryLittleEndian<float>(&binary_file, point.z);

    WriteBinaryLittleEndian<uint8_t>(&binary_file, point.r);
    WriteBinaryLittleEndian<uint8_t>(&binary_file, point.g);
    WriteBinaryLittleEndian<uint8_t>(&binary_file, point.b);
  }

  binary_file.close();
}

}  // namespace PSL
