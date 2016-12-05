# offt
Maryland's auto-tuned parallel FFT algorithm.

# You can run it on Hopper or Edison at NERSC.

# compile the Active Harmony
cd activeharmony
make
cd ..

# compile the FFT code
# There will be a compile warning about using 'gethostbyname', but you can ignore it.
# You can also link the 2decomp&fft library and the p3dfft library with run-fft.c by editing Makefile.
module load fftw
make

# execute a test run
qsub job-test.sh

# remove temp files
# The test run will generate temporary files named tmp-db-* and tmp-uv-*.
# You can manually remove the files after the test run.
rm -f tmp-db-* tmp-uv-*
