#!/bin/bash
set -e

cmake -B build
cmake --build build
build/cpptest
