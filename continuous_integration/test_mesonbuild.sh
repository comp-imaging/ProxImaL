#!/bin/bash

set -e
cd proximal/halide
meson build
ninja -C build
