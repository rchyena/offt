#
# OFFT specific setup.
#
SRCS=run-fft.c offt-compute.c offt-tuning.c

#
# Dependency setup.
#
ifndef HARMONY_HOME
  DEPS += activeharmony-4.6.0
  HARMONY_HOME = ./activeharmony-4.6.0
endif

#
# Build flag setup.
#
override CFLAGS  += -DA2AV -DSTRIDE -I$(HARMONY_HOME)/include
override LDFLAGS += -L$(HARMONY_HOME)
override LDLIBS  += -lharmony

BUG_CC=/usr/local/stow/openmpi-1.4.1-gm/bin/mpicxx
BUG_COPTS=-O2
#P3DFFT_OPTS=-DGNU -DONED -DESTIMATE -DMEASURE -DUSE_EVEN -DSTRIDE1 -DFFTW
BUG_P3DFFT_OPTS=-DGNU
HOP_P3DFFT_OPTS=-DPGI
HOP_COPTS=-fast # O3...
EDS_COPTS=-O3 # O3...

.PHONY: deps harmony distclean clean

#
# Target/recipe definitions.
#

#all: bug
all: hop
#all: eds

$(SRCS:.c=.o): deps

test:
	mpicc $(P3DFFT_OPTS) -I ~/local/p3dfft/include -I /usr/local/stow/openmpi-1.4.1-gm/include -I ~/local/fftw-3.3.2-gm/include -c -o test.o test.c
	mpicc -L /usr/local/stow/openmpi-1.4.1-gm/lib -o test test.o ~/package/p3dfft.2.5.1/build/libp3dfft.a -L ~/local/fftw-3.3.2-gm/lib -lfftw3_mpi -lfftw3 -lgfortran -lmpi_f90 -lmpi_f77

bug:
	$(BUG_CC) -DA2AV -DSTRIDE $(BUG_COPTS) -I /usr/local/stow/openmpi-1.4.1-gm/include -I ~/local/fftw-3.3.2-gm/include -I ~/local/libNBC-1.1.1-gm/include -c -o offt-compute.o offt-compute.c -Wall
	$(BUG_CC) -DA2AV -DSTRIDE $(BUG_P3DFFT_OPTS) $(BUG_COPTS) -I ~/local/p3dfft/include -I /usr/local/stow/openmpi-1.4.1-gm/include -I ~/local/fftw-3.3.2-gm/include -c -o run-fft.o run-fft.c -Wall
	$(BUG_CC) -DA2AV -DSTRIDE $(BUG_COPTS) -I $(HARMONY_HOME)/include -I /usr/local/stow/openmpi-1.4.1-gm/include -I ~/local/fftw-3.3.2-gm/include -c -o offt-tuning.o offt-tuning.c -Wall
	$(BUG_CC) $(BUG_COPTS) -o run-fft offt-compute.o offt-tuning.o run-fft.o -L ~/local/p3dfft/lib -lp3dfft -L ~/local/fftw-3.3.2-gm/lib -lfftw3_mpi -lfftw3 -L /usr/local/stow/openmpi-1.4.1-gm/lib -L ~/local/libNBC-1.1.1-gm/lib -lnbc $(HARMONY_HOME)/lib/libharmony.a -Wall -lgfortran -lmpi_f90 -lmpi_f77

hop:
	cc -DA2AV -DSTRIDE $(HOP_COPTS) -DSHSONG_HOPPER -DSHSONG_MPICH -c -o offt-compute.o offt-compute.c
	cc -DA2AV -DSTRIDE $(HOP_COPTS) -DSHSONG_HOPPER -DSHSONG_MPICH -c -o run-fft.o run-fft.c
	cc -DA2AV -DSTRIDE $(HOP_COPTS) -DSHSONG_HOPPER -DSHSONG_MPICH -I $(HARMONY_HOME)/include -c -o offt-tuning.o offt-tuning.c
	cc $(HOP_COPTS) -o run-fft offt-compute.o offt-tuning.o run-fft.o $(EDS_AH_HOME)/lib/libharmony.a
	#cc -DA2AV -DSTRIDE $(HOP_COPTS) $(HOP_P3DFFT_OPTS) -DSHSONG_HOPPER -DSHSONG_MPICH -I ~/local/p3dfft/include -c -o run-fft.o run-fft.c
	#cc -DA2AV -DSTRIDE $(HOP_COPTS) -DSHSONG_HOPPER -DSHSONG_MPICH -I $(HARMONY_HOME)/include -c -o offt-tuning.o offt-tuning.c
	#cc $(HOP_COPTS) -o run-fft offt-compute.o offt-tuning.o run-fft.o -L ~/local/p3dfft/lib -lp3dfft $(HARMONY_HOME)/lib/libharmony.a ~/package/hopper.2decomp_fft/lib/lib2decomp_fft.a

eds:
	cc -DA2AV -DSTRIDE $(EDS_COPTS) -DSHSONG_EDISON -DSHSONG_MPICH -c -o offt-compute.o offt-compute.c
	cc -DA2AV -DSTRIDE $(EDS_COPTS) -DSHSONG_EDISON -DSHSONG_MPICH -c -o run-fft.o run-fft.c
	cc -DA2AV -DSTRIDE $(EDS_COPTS) -DSHSONG_EDISON -DSHSONG_MPICH -I $(HARMONY_HOME)/include -c -o offt-tuning.o offt-tuning.c
	cc $(EDS_COPTS) -o run-fft offt-compute.o offt-tuning.o run-fft.o $(HARMONY_HOME)/lib/libharmony.a
	#cc $(EDS_COPTS) -o run-fft offt-compute.o offt-tuning.o run-fft.o $(HARMONY_HOME)/lib/libharmony.a ~/package/edison.2decomp_fft/lib/lib2decomp_fft.a

deps: $(filter-out $(wildcard $(DEPS)), $(DEPS))

activeharmony-4.6.0: ah-4.6.0.tar.gz
	tar -zxf $<
	$(MAKE) -C $@ install

ah-4.6.0.tar.gz:
	wget --no-verbose http://www.dyninst.org/sites/default/files/downloads/harmony/ah-4.6.0.tar.gz

clean:
	rm -f *.o run-fft

distclean: clean
	rm -rf *~ activeharmony-4.6.0 ah-4.6.0.tar.gz
