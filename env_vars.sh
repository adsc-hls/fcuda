#!/bin/bash
#Please source this file before testing any benchmark

#Path to FCUDA directory. Modify it according to
#where you put FCUDA in your system
export FCUDA_DIR=
#Path to Xilinx software
export VIVADO_HLS=
export VIVADO=
export XILINX_SDK=

export FROOT=$FCUDA_DIR/fcuda_src
export PATH=$FROOT/bin:$VIVADO/bin:$VIVADO_HLS/bin:$XILINX_SDK/bin:$XILINX_SDK/gnu/arm/lin/bin:$PATH

export BENCHMARKS=$FCUDA_DIR/fcuda-benchmarks
#List of benchmarks
export CP=$BENCHMARKS/cp
export DWT=$BENCHMARKS/dwt
export FWT=$BENCHMARKS/fwt
export MATMUL=$BENCHMARKS/matmul
export HOTSPOT_float=$BENCHMARKS/hotspot
export HOTSPOT_int=$BENCHMARKS/hotspot/int
export PATHFINDER=$BENCHMARKS/pathfinder
export LAVAMD_float=$BENCHMARKS/lavaMD
export LAVAMD_int=$BENCHMARKS/lavaMD/int
export STREAMCLUSTER=$BENCHMARKS/streamcluster
export GAUSSIAN=$BENCHMARKS/gaussian
export NN=$BENCHMARKS/nn
export NW=$BENCHMARKS/nw
export LUD=$BENCHMARKS/lud
export BACKPROP=$BENCHMARKS/backprop
export BFS=$BENCHMARKS/bfs
export PARTICLEFILTER_NAIVE=$BENCHMARKS/particlefilter_naive
export SRAD_V2=$BENCHMARKS/srad_v2
export CONV1D=$BENCHMARKS/conv1d
export DCT=$BENCHMARKS/dct
export IDCT=$BENCHMARKS/idct




