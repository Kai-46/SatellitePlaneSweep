// This file is part of PlaneSweepLib (PSL)

// Copyright 2016 Christian Haene (ETH Zuerich)

// PSL is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// PSL is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with PSL.  If not, see <http://www.gnu.org/licenses/>.

#ifndef GRID_H
#define GRID_H

#include <string>
#include <fstream>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <limits.h>
#include <assert.h>

#include "ioTools.h"

namespace PSL {

// 3D grid data on CPU
// it has the capability to io with disk
template<typename T>
class Grid {
 public:
  Grid();

  Grid(int xDim, int yDim, int zDim, T value);

  ~Grid();

  inline const T &operator()(int x, int y, int z = 0) const;

  inline T &operator()(int x, int y, int z = 0);

  inline const T &operator()(int idx) const;

  inline T &operator()(int idx);

  void clone(Grid<T> &destination) const;

  int getWidth() const;

  int getHeight() const;

  int getDepth() const;

  int getNbVoxels() const;

  void freeMem();

  T *getDataPtr(int z=0) const;

  void resize(int xDim, int yDim, int zDim);

  void saveAsDataFile(const std::string &fileName) const;

  void loadFromDataFile(const std::string &fileName);

 protected:
  T *_cells;
  int _xDim, _yDim, _zDim;

  int _xyDim;
  int _xyzDim;
};

template<typename T>
Grid<T>::Grid() {
  _xDim = 0;
  _yDim = 0;
  _zDim = 0;
  _xyDim = 0;
  _xyzDim = 0;
  _cells = NULL;
}

template<typename T>
Grid<T>::Grid(int xDim, int yDim, int zDim, T value) {
  _xDim = xDim;
  _yDim = yDim;
  _zDim = zDim;
  _xyDim = xDim * yDim;
  _xyzDim = _xyDim * zDim;
  _cells = new T[_xyzDim];
  std::fill(_cells, _cells + _xyzDim, value);
}

template<typename T>
Grid<T>::~Grid() {
  freeMem();
}

template<typename T>
void Grid<T>::freeMem() {
  if (_cells) {
    delete [] _cells;
    _cells = NULL;
  }
}

template<typename T>
inline const T &Grid<T>::operator()(int x, int y, int z) const {
  return _cells[z * _xyDim + y * _xDim + x];
}

template<typename T>
inline T &Grid<T>::operator()(int x, int y, int z) {
  return _cells[z * _xyDim + y * _xDim + x];
}

template<typename T>
inline const T &Grid<T>::operator()(int idx) const {
  return _cells[idx];
}

template<typename T>
inline T &Grid<T>::operator()(int idx) {
  return _cells[idx];
}

template<typename T>
void Grid<T>::resize(int xDim, int yDim, int zDim) {
  if (xDim == _xDim && yDim == _yDim && zDim == _zDim) {
    return;
  }

  _xDim = xDim;
  _yDim = yDim;
  _zDim = zDim;
  _xyDim = xDim * yDim;
  _xyzDim = _xyDim * zDim;

  assert (_xyzDim > 0);    // allocating a huge memory out of the range of int32 is not supported

  freeMem();
  _cells = new T[_xyzDim];
}

template<typename T>
int Grid<T>::getWidth() const {
  return _xDim;
}

template<typename T>
int Grid<T>::getHeight() const {
  return _yDim;
}

template<typename T>
int Grid<T>::getDepth() const {
  return _zDim;
}

template<typename T>
int Grid<T>::getNbVoxels() const {
  return _xyzDim;
}

template<typename T>
T *Grid<T>::getDataPtr(int Z) const {
  return _cells + Z * _xyDim;
}

template<typename T>
void Grid<T>::clone(Grid<T> &destination) const {
  destination.resize(this->getWidth(), this->getHeight(), this->getDepth());
  std::memcpy(destination.getDataPtr(), this->getDataPtr(), sizeof(T) * this->getNbVoxels());
}

template<typename T>
void Grid<T>::saveAsDataFile(const std::string &fileName) const {
  std::ofstream outStream;
  outStream.open(fileName.c_str(), std::ios::out | std::ios::binary);

  if (!outStream.is_open()) {
    throw std::runtime_error("Could not open grid data output file for writing.");
  }

  // file format version, might be useful at some point
  unsigned char version = 1;
  outStream.write((char *) &version, 1);

  // endianness
  unsigned char endian = is_little_endian() ? 0 : 1;
  outStream.write((char *) &endian, 1);

  // store sizes of data types written
  // first int in an unsigned char because we know that char has always size 1
  unsigned char intSize = sizeof(int);
  outStream.write((char *) &intSize, 1);

  // treat the data type T as int
  int elemSize = sizeof(T);
  outStream.write((char *) &elemSize, sizeof(int));

  // now we store the size of the grid
  outStream.write((char *) &_xDim, sizeof(int));
  outStream.write((char *) &_yDim, sizeof(int));
  outStream.write((char *) &_zDim, sizeof(int));

  // now grid data is written
  outStream.write((char *) this->getDataPtr(), elemSize * _xyzDim);

  if (!outStream.good()) {
    throw std::runtime_error("An error occured while writing the grid to a data file.");
  }

  outStream.close();
}

template<typename T>
void Grid<T>::loadFromDataFile(const std::string &fileName) {
  std::ifstream inStream;
  inStream.open(fileName.c_str(), std::ios::in | std::ios::binary);

  if (!inStream.is_open()) {
    throw std::runtime_error("Could not open grid data input file.");
  }

  // read in version
  unsigned char version;
  inStream.read((char *) &version, 1);
  if (version != 1) {
    throw std::runtime_error("Only version 1 is supported.");
  }

  // read in endian
  unsigned char endian;
  inStream.read((char *) &endian, 1);

  unsigned char currentEndian = is_little_endian() ? 0 : 1;
  if (endian != currentEndian) {
    throw std::runtime_error("Current platform does not have the same endian as the depht map data file.");
  }

  // read in the size of an int from file
  unsigned char intSize;
  inStream.read((char *) &intSize, 1);

  // check if current plattform has the same int size
  if (intSize != sizeof(int)) {
    throw std::runtime_error(
        "Current platform does not have the same int size as the one the file was written with.");
  }

  int elemSize;
  inStream.read((char *) &elemSize, sizeof(int));
  if (elemSize != sizeof(T)) {
    throw std::runtime_error(
        "Size of the datatype stored in the grid does not match with the one from the file.");
  }

  // read the grid size
  int width, height, depth;
  inStream.read((char *) &width, sizeof(int));
  inStream.read((char *) &height, sizeof(int));
  inStream.read((char *) &depth, sizeof(int));

  // resize the grid
  this->resize(width, height, depth);

  // load the data stored in the grid
  inStream.read((char *) this->getDataPtr(), sizeof(T) * width * height * depth);

  // check stream
  if (!inStream.good()) {
    throw std::runtime_error("Error while loading the grid from the data file");
  }

  inStream.close();
}

}  // namespace PSL

#endif // GRID_H
