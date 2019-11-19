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

#include "ioTools.h"
#include <fstream>
#include <psl_base/exception.h>
#include <boost/filesystem.hpp>
#include <iomanip>

namespace PSL {

bool is_little_endian() {
  short int word = 1;
  char *byte = (char *) &word;
  return byte[0] != 0;
}

std::string extractBaseFileName(const std::string &fullFileName) {
  size_t pos = fullFileName.find_last_of('/');
  std::string baseName;
  if (pos != std::string::npos) {
    baseName = fullFileName.substr(pos + 1, fullFileName.size() - pos);
  } else {
    baseName = fullFileName;
  }
  // remove the ending
  pos = baseName.find_first_of('.');
  if (pos != std::string::npos) {
    return baseName.substr(0, pos);
  } else {
    return baseName;
  }
}

std::string extractFileName(const std::string &fullFileName) {
  size_t pos = fullFileName.find_last_of('/');
  if (pos != std::string::npos) {
    return fullFileName.substr(pos + 1, fullFileName.size() - pos);
  } else {
    return fullFileName;
  }
}

std::string extractPath(const std::string &fullFileName) {
  size_t pos = fullFileName.find_last_of('/');
  if (pos != std::string::npos) {
    return fullFileName.substr(0, pos + 1);
  } else {
    return "./";
  }
}

void makeOutputFolder(const std::string &folderName) {
  if (!boost::filesystem::exists(folderName)) {
    if (!boost::filesystem::create_directory(folderName)) {
      std::stringstream errorMsg;
      errorMsg << "Could not create output directory: " << folderName;
      PSL_THROW_EXCEPTION(errorMsg.str().c_str());
    }
  }
}

std::string toFixedWidthString(int value, int fixed_width) {
  std::ostringstream oss;
  oss << std::setw(fixed_width) << std::setfill('0') << value;

  return oss.str();
}

}   // namespace PSL
