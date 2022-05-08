# SOFM - Self-organizing Feature Maps module in C++ with Python wrappings

## Install

### C++

Requires `cmake`, c++ compiler and Eigen. All can probably be installed with (on Ubuntu 18.04)
```{bash}
sudo apt install cmake build-essential libeigen3-dev
```

### Python wrappings

The pybind11 folder comes with the rest of the code, so all you should need to do is
```{bash}
pip install .
```
to install `sofm` on the system.