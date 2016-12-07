#PBS -j oe
#PBS -l walltime=00:05:00
#PBS -q debug
#PBS -l mppwidth=72
#PBS -l mppnppn=24
#!/bin/bash
cd $PBS_O_WORKDIR
#FFTW
aprun -n 64 -N 24 ./run-fft -N 512 -n 512 -L 512 -r 5 -m 1 -v -a 1
#OFFT
export HARMONY_S_HOST=localhost
export HARMONY_S_PORT=1900
aprun -n 64 -N 24 ./run-fft -N 512 -n 512 -L 512 -r 5 -m 1 -v -a 0 -o -e -A 4 -l 200 -O 0
