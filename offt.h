/*
 * Copyright 2016 Jeffrey K. Hollingsworth
 *
 * This file is part of OFFT, University of Maryland's auto-tuned
 * parallel FFT algorithm.
 *
 * OFFT is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * OFFT is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
 * or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with OFFT.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef OFFT_INCLUDE
#define OFFT_INCLUDE

#define SUBTILE_SIZE (8192) /* 256KB cache, source/dest */
#define ADJUST_POINT /* to avoid low values for the 1-D cases */
//#define ROTATE_RANKORDER /* rotate core mapping order */
#define NOTEST /* in case asyn msg progress is supported */
//#define FINETEST
#define TUNING_REPS 1
#define BUFMALLOC
//#define STRIDE
//#define A2AV
// STRIDE: YX_STRIDE_XYZ
// malloc for input and buffer (Nxxm2xNz vs NxxNyxm3)
// TRANSPOSE: zyx
// TRANSPOSE: yzx (Nx=Ny)

#define P1D_LIST
#define FAST_TUNING
#define TRANSPOSE_LIST
#define AH_TUNING

#ifdef SHSONG_HOPPER
#define HOME_DIR "."
#elif defined(SHSONG_EDISON)
#define HOME_DIR "."
#else
#define HOME_DIR "."
#endif

#define BUFFER_SIZE_LIMIT (32*1024*1024) /* 512MB: 32M elements */

#if defined(SHSONG_HOPPER) || defined(SHSONG_EDISON)
#else
#define AVOID_TILE
// to avoid a weird behavior of openmpi of BUG cluster
// When MPI_Alloc_mem is used for a specific-sized buffer,
//   MPI_Wait blocks forever after A2A.
// If MPI_Alloc_mem is replaced with new and delete, it comes back to normal.
// But we have to use MPI_Alloc_mem for better performance of myrinet.
#endif

#include "fftw3.h"
#include "fftw3-mpi.h"

/* **********************************************************
 @ structure for parameters
   ********************************************************** */
struct _offt_params {
  int is_converged;  /* 0:tuning not finished  1:finished */
  int is_infeasible;  /* 0:params in a feasible area  1:infeasible area */
  int is_in_database;  /* 0:params not in a database file  1:in a database file */
#define LOG0 (-1)
#define _P1_ 0  /* decomposition factor p1: # processes on x dim (p = p1 x p2) */
#define _T1_ 1  /* tile size in phase 1: # elements on x dim */
#define _W1_ 2  /* window size in phase 1: max # tiles whose comm. exist on at the same time */
#define _Px1_ 3  /* packing sub-tile size on x dim in phase 1: # elements on x dim */
#define _Py1_ 4  /* packing sub-tile size on y dim in phase 1: # elements on y dim */
#define _Fz_ 5  /* MPI_Test call frequency during FFTz in phase 1: total # calls during FFTz on one tile */
#define _FP1_ 6  /* MPI_Test call frequency during Pack in phase 1: total # calls during Pack on one tile */
#define _Ux1_ 7  /* unpacking sub-tile size on x dim in phase 1: # elements on x dim */
#define _Uz1_ 8  /* unpacking sub-tile size on z dim in phase 1: # elements on z dim */
#define _FU1_ 9  /* MPI_Test call frequency during Unpack in phase 1: total # calls during Unpack on one tile */
#define _Fy1_ 10  /* MPI_Test call frequency during FFTy in phase 1: total # calls during FFTy on one tile */
#define _Ry_ 11  /* ratio of #ffty in phase I to total#ffty (0-10) */
#define _T2_ 12  /* tile size in phase 2: # elements on z dim */
#define _W2_ 13  /* window size in phase 2: max # tiles whose comm. exist on at the same time */
#define _Pz2_ 14  /* packing sub-tile size on z dim in phase 2: # elements on z dim */
#define _Px2_ 15  /* packing sub-tile size on x dim in phase 2: # elements on x dim */
#define _Fy2_ 16  /* MPI_Test call frequency during FFTy in phase 2: total # calls during FFTy on one tile */
#define _FP2_ 17  /* MPI_Test call frequency during Pack in phase 2: total # calls during Pack on one tile */
#define _Uz2_ 18  /* unpacking sub-tile size on z dim in phase 2: # elements on z dim */
#define _Uy2_ 19  /* unpacking sub-tile size on y dim in phase 2: # elements on y dim */
#define _FU2_ 20  /* MPI_Test call frequency during Unpack in phase 2: total # calls during Unpack on one tile */
#define _Fx_ 21  /* MPI_Test call frequency during FFTx in phase 2: total # calls during FFTx on one tile */
#define _V_ 22  /* 2-bit switch for A2AV leftbit:phase0, rightbit:phase1 */
#define _S_ 23  /* switch for 1-D FFT method 0:TRANSPOSE 1:STRIDE */
#define PARAM_COUNT 24
  int v[PARAM_COUNT];  /* array for parameter values */
};

struct _offt_comm {
  int p1;
  int p2;
  MPI_Comm *comm1;
  MPI_Comm *comm2;
  MPI_Group *group1;
  MPI_Group *group2;
#ifdef A2AV
  int M1; /* ceil(Nx/p1) */
  int M2; /* ceil(Ny/p2) */
  int M3; /* ceil(Nz_new/p2) */
  int M4; /* ceil(Ny/p1) */
  int F1; /* floor(Nx/p1) */
  int F2; /* floor(Ny/p2) */
  int F3; /* floor(Nz_new/p2) */
  int F4; /* floor(Ny/p1) */
  int m1; /* # my elements on x */
  int m2; /* # my elements on y during A2A1 */
  int m3; /* # my elements on z */
  int m4; /* # my elements on y during A2A2 */
  int b1; /* # over-loaded nodes with floor(Nx/p1)+1 */
  int b2; /* # over-loaded nodes with floor(Ny/p2)+1 */
  int b3; /* # over-loaded nodes with floor(Nz_new/p2)+1 */
  int b4; /* # over-loaded nodes with floor(Ny/p1)+1 */
#else
  int m1; //  my # of elements in x dim at the beginning
  int M1; // max # of elements in x dim at the beginning
  int m2; //  my # of elements in y dim at the beginning
  int M2; // max # of elements in y dim at the beginning
  int m3; //  my # of elements in z dim after one A2A
  int M3; // max # of elements in z dim after one A2A
  int m4; //  my # of elements in y dim after two A2A
  int M4; // max # of elements in y dim after two A2A
#endif
  int istart[3]; /* starting coodinates x,y,z */
  int isize[3]; /* # elements on each dimenstion */
  int istride[3]; /* memory stride amount */
  int ostart[3];
  int osize[3];
  int ostride[3];
};

struct _offt_plan {
  /* parameter-independent settings */
  int p;
  int rank;
  int Nx;
  int Ny;
  int Nz;
  int is_r2c;
  int fftw_flag;
  int ah_strategy;
  int max_loop;
  int tuning_mode; /* 0:p1xp2 1:1xp 2:px1 */
  int is_W0; /* W1=W2=0 */
  int extrapolation_window; /* extrapolation window size for fast tuning */
  int is_oned; /* 1-D decomposition for 1xp or px1 cases */
  int is_a2a; /* use MPI_Alltoall when W = 0 */
  int is_equalxy; /* use the optimization method when Nx == Ny, output memory layout will be yzx */
#ifdef NOTEST
  int is_notest; /* set all frequency values to be zero */
#endif
#define INIT_ALL 0
#define INIT_FFTW 1
#define INIT_AH 2
#define INIT_BUFFER 3
#define T_INIT_COUNT 4
  double t_init[T_INIT_COUNT];
/* the timer array components */
#define ALL 0
#define INIT1 1
#define WAIT1 2
#define TEST1 3
#define INIT2 4
#define WAIT2 5
#define TEST2 6
#define FFTz 7
#define FFTy1 8
#define FFTy2 9
#define FFTx 10
#define TRANSPOSE 11
#define PACK1 12
#define UNPACK1 13
#define PACK2 14
#define UNPACK2 15
#define GES 16
  double t[GES];
#ifdef AVOID_TILE
  int bad_tile_count;
#define MAX_BAD_TILE_COUNT 0
#if 0
//#define ERROR_D (640 * 640 / 64 * 640 * 16) // 65536000
#define ERROR_1 (1024 * 1024 / 32 * 1024 * 2) // 67108864
#define ERROR_2 (768 * 768 / 32 * 768 * 8) // 113246208
#define ERROR_3 (640 * 640 / 64 * 640 * 32) // 131072000
#define ERROR_LOW ERROR_1
#define ERROR_HIGH ERROR_3
#else
#define ERROR_1 (2 * 512/16 * 512 / 16) // N=640 p=1x16 T1=2
//#define ERROR_1 (2 * 640/16 * 640) // N=640 p=1x16 T1=2
#define ERROR_2 (8 * 640/32 * 640 / 32) // N=640 p=1x32 T1=8
#define ERROR_LOW ERROR_1
#define ERROR_HIGH ERROR_2
#endif
#endif
  char point_database_file[256];
  char user_vertex_file[256];

  /* parameter-dependent settings */
  struct _offt_params *params;
  struct _offt_comm *comm;
#ifdef BUFMALLOC
  void* buffer_chunk; //
#endif
  void* buffers1; //struct _offt_buffer** buffers
  void* buffers2; //struct _offt_buffer** buffers
  fftw_plan pt_transpose; // 3-d in-place tile transpose
#ifdef TRANSPOSE_LIST
  fftw_plan* pt_transpose_list; // 3-d in-place tile transpose
  int pt_transpose_list_size;
#endif
  fftw_plan p1d_x; /* 1-D in-place transform */
  fftw_plan p1d_y; /* 1-D in-place transform */
  fftw_plan p1d_z; /* 1-D in-place transform */
  fftw_plan p1d_x_t; /* only for transpose */
  fftw_plan p1d_y_t; /* only for transpose */
#ifdef P1D_LIST
  fftw_plan* p1d_x_s_list; /* only for stride */
  fftw_plan* p1d_y_s_list; /* only for stride */
  int p1d_xy_s_list_size; /* only for stride */
#endif
};

#ifdef NOTEST
struct _offt_plan* offt_3d_init(int Nx, int Ny, int Nz, double* in, double* out, int is_r2c, int fftw_flag, int is_oned, int is_a2a, int is_equalxy, int is_notest, int ah_strategy, int max_loop, int tuning_mode, int is_W0, int extrapolation_window, struct _offt_params *custom_params);
#else
struct _offt_plan* offt_3d_init(int Nx, int Ny, int Nz, double* in, double* out, int is_r2c, int fftw_flag, int is_oned, int is_a2a, int is_equalxy, int ah_strategy, int max_loop, int tuning_mode, int is_W0, int extrapolation_window, struct _offt_params *custom_params);
#endif
void offt_3d_fin(struct _offt_plan *po);
void offt_3d_execute(struct _offt_plan *po, double* in, double* out, int is_tuning);
//void print_complex(double *a, int n, int stride);
void print_params(int *v);
void offt_print_time(double *t);

/* restore inline behavior for non-gcc compilers */
#ifndef __GNUC__
#define __inline__ inline 
#endif

static __inline__ int max(int a, int b) {
  return (a > b)?a:b;
}

static __inline__ int min(int a, int b) {
  return (a < b)?a:b;
}

#endif // OFFT_INCLUDE
