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

//#define P3DFFT
//#define DECOMP

#ifdef P3DFFT
#ifdef SHSONG_MPICH
#include "p3dfft.h"
#else
#include "p3dfft-cpp.h" /* use c++ compiler for libNBC */
#endif
#endif
#ifdef DECOMP
#include "2decomp.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <mpi.h>
#include "offt.h"
#include "fftw3.h"
#include "fftw3-mpi.h"

//#define RAND_INPUT

void init(int fft_alg, int Nx, int Ny, int Nz, int p1, int p2, int rank, int is_r2c, double *in, double *out, struct _offt_plan* po) {
  int x, y, z;
  if (fft_alg == 0) {
    struct _offt_comm* comm = po->comm;
    for(x=0; x<comm->isize[0]; x++) {
      for(y=0; y<comm->isize[1]; y++) {
        for(z=0; z<comm->isize[2]; z++) {
          if (is_r2c) {
            in[0 + 1*(z+2*comm->istride[1]*y+2*comm->istride[0]*x)] = z+10*(y+comm->istart[1])+100*(x+comm->istart[0]);
          } else {
            in[0 + 2*(z+comm->istride[1]*y+comm->istride[0]*x)] = z+10*(y+comm->istart[1])+100*(x+comm->istart[0]);
            in[1 + 2*(z+comm->istride[1]*y+comm->istride[0]*x)] = 0.0;
          }
        }
      }
    }
  } else if (fft_alg == 1) {
    int Nz_new = (is_r2c)?Nz/2+1:Nz;
    int M1 = (Nx+p1-1)/p1;
    int x_off = M1 * rank;
    for(x=0; x<min(M1, Nx-x_off); x++) {
      for(y=0; y<Ny; y++) {
        for(z=0; z<Nz; z++) {
          if (is_r2c) {
            in[0 + 1*(z+2*Nz_new*(y+Ny*x))] = z+10*y+100*(x+x_off);
          } else {
            in[0 + 2*(z+Nz_new*(y+Ny*x))] = z+10*y+100*(x+x_off);
            in[1 + 2*(z+Nz_new*(y+Ny*x))] = 0.0;
          }
        }
      }
    }
#ifdef P3DFFT
  } else if (fft_alg == 2) {
    int conf;
    int istart[3], iend[3], isize[3];
    conf = 1;
    p3dfft_get_dims(istart, iend, isize, conf);
    //printf("%d: istart %d %d %d iend %d %d %d isize %d %d %d\n",
    //  rank, istart[0], istart[1], istart[2], iend[0], iend[1], iend[2], isize[0], isize[1], isize[2]);
    for(x=0; x<iend[2]-istart[2]+1; x++) {
      for(y=0; y<iend[1]-istart[1]+1; y++) {
        for(z=0; z<Nz; z++) {
          /* successive */
#ifdef RAND_INPUT
          *in++ = (double)rand() / RAND_MAX;
#else
          *in++ = z+10*(y+istart[1]-1)+100*(x+istart[2]-1);
#endif
        }
      }
    } /* end of for x */
#endif
#ifdef DECOMP
  } else if (fft_alg == 3) {
    /* TODO: Nx%p1 != 0 */
    int new_Nz = (is_r2c)?Nz/2+1:Nz;
    int M1 = (Nx+p1-1)/p1;
    int M2 = (Ny+p2-1)/p2;
    int M3 = (new_Nz+p2-1)/p2;
    int M4 = (Ny+p1-1)/p1;
    int x_off = M1 * (rank / p2);
    int y_off = M2 * (rank % p2);
    int yz_plane_size = (M2*p2 > M4*p1)? M2*M3*p2: M3*M4*p1;
    for(x=0; x<min(M1, Nx-x_off); x++) {
      for(y=0; y<min(M2, Ny-y_off); y++) {
        for(z=0; z<Nz; z++) {
          if (is_r2c) {
            in[0 + 1*(z+2*(M3*p2)*y+2*yz_plane_size*x)] = z+10*(y+y_off)+100*(x+x_off);
          } else {
            in[0 + 2*(z+(M3*p2)*y+yz_plane_size*x)] = z+10*(y+y_off)+100*(x+x_off);
            //in[1 + 2*(z+(M3*p2)*y+yz_plane_size*x)] = 10000+z+10*(y+y_off)+100*(x+x_off);
            in[1 + 2*(z+(M3*p2)*y+yz_plane_size*x)] = 0.0;
          }
        }
      }
    } /* end of for x */
#endif
  }
  return;
}

int main(int argc, char** argv) {
  int Nx = 32, Ny = 32, Nz = 32;
  int p1 = -1;
  int reps = 1;
  int p;
  int rank;
  //double *in;
  double *out;
  double t_min = 999999999.0;
  double offt_t_min[GES];
  int fftw_tuning_level = 0;
  unsigned int fftw_flag = 0;
  int ah_strategy = 0;
  int max_loop = 0;
  int tuning_mode = 0;
  int is_W0 = 0;
  int extrapolation_window = 0;
  int is_oned = 0;
  int is_a2a = 0;
  int is_equalxy = 0;
#ifdef NOTEST
  int is_notest = 0;
#endif
  int fft_alg = 0;
  int verbose = 0;
  int is_r2c = 0;
  //double reduce_t = 0.0;
  double t = 0.0;
  fftw_plan mpi_p3d = NULL;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &p);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  /* custom parameters for fft_alg == 0 */
  struct _offt_params *custom_params =
    (struct _offt_params*)malloc(sizeof(struct _offt_params));
  int i;
  for (i = 0; i < PARAM_COUNT; i++)
    custom_params->v[i] = -1;

  /* handle input arguments */
#ifdef NOTEST
  char *optchars = "N:n:L:va:Rr:m:A:O:Qobecs:l:d:T:W:t:w:P:p:X:Y:Z:U:u:y:z:E:F:f:G:H:h:I:i:V:S:", c;
#else
  char *optchars = "N:n:L:va:Rr:m:A:O:Qobes:l:d:T:W:t:w:P:p:X:Y:Z:U:u:y:z:E:F:f:G:H:h:I:i:V:S:", c;
#endif
  extern char *optarg;
  while ( (c = getopt(argc, argv, optchars)) >= 0 ) {
    switch (c) {
    case 'N': Nx = atoi(optarg); break;
    case 'n': Ny = atoi(optarg); break;
    case 'L': Nz = atoi(optarg); break;
    case 'v': verbose = 1; break;
    case 'a': fft_alg = atoi(optarg); break;
    case 'R': is_r2c = 1; break;
    case 'r': reps = atoi(optarg); break;
    case 'm': /* for fft_alg 0 or 1 */
      fftw_tuning_level = atoi(optarg);
      switch (fftw_tuning_level % 4) {
        case 0: fftw_flag = FFTW_ESTIMATE; break;
        case 1: fftw_flag = FFTW_MEASURE; break;
        case 2: fftw_flag = FFTW_PATIENT; break;
        case 3: fftw_flag = FFTW_EXHAUSTIVE; break;
      }
      break;
     /* for fft_alg 0 */
    case 'o': is_oned = 1; break; /* use 1-D method for px1 or 1xp case */
    case 'b': is_a2a = 1; break;
    case 'e': is_equalxy = 1; break;
#ifdef NOTEST
    case 'c': is_notest = 1; break;
#endif
    case 's': ah_strategy = atoi(optarg); break;
    case 'l': max_loop = atoi(optarg); break;
    case 'O': tuning_mode = atoi(optarg); break; /* 0:p1xp2 1:1xp 2:px1 */
    case 'Q': is_W0 = 1; break; /* set W1=W2=0 */
    case 'A': extrapolation_window = atoi(optarg); break; /* use extrapolation techniuqe for fast tuning */
    /* parameters */
    case 'd': p1 = custom_params->v[_P1_] = atoi(optarg); break;
    case 'T': custom_params->v[_T1_] = atoi(optarg); break;
    case 'W': custom_params->v[_W1_] = atoi(optarg); break;
    case 't': custom_params->v[_T2_] = atoi(optarg); break;
    case 'w': custom_params->v[_W2_] = atoi(optarg); break;
    case 'Y': custom_params->v[_Ry_] = atoi(optarg); break;
    case 'P': custom_params->v[_Px1_] = atoi(optarg); break;
    case 'p': custom_params->v[_Py1_] = atoi(optarg); break;
    case 'X': custom_params->v[_Px2_] = atoi(optarg); break;
    case 'Z': custom_params->v[_Pz2_] = atoi(optarg); break;
    case 'U': custom_params->v[_Ux1_] = atoi(optarg); break;
    case 'u': custom_params->v[_Uz1_] = atoi(optarg); break;
    case 'y': custom_params->v[_Uy2_] = atoi(optarg); break;
    case 'z': custom_params->v[_Uz2_] = atoi(optarg); break;
    case 'E': custom_params->v[_Fz_] = atoi(optarg); break;
    case 'F': custom_params->v[_Fy1_] = atoi(optarg); break;
    case 'f': custom_params->v[_Fy2_] = atoi(optarg); break;
    case 'G': custom_params->v[_Fx_] = atoi(optarg); break;
    case 'H': custom_params->v[_FP1_] = atoi(optarg); break;
    case 'h': custom_params->v[_FP2_] = atoi(optarg); break;
    case 'I': custom_params->v[_FU1_] = atoi(optarg); break;
    case 'i': custom_params->v[_FU2_] = atoi(optarg); break;
    case 'V': custom_params->v[_V_] = atoi(optarg); break;
    case 'S': custom_params->v[_S_] = atoi(optarg); break;
    }
  }

  /* print input arguments and parameters */
  if (!rank) {
    int a;
    for (a = 0; a < argc; a++) {
      printf("%s ", argv[a]);
    }
    printf("\n");
#ifdef NOTEST
    printf("Nx %d Ny %d Nz %d p %d p1 %d r %d a %d m %d o %d b %d e %d c %d s %d l %d O %d Q %d A %d\n",
      Nx, Ny, Nz, p, p1, reps, fft_alg, fftw_tuning_level, is_oned, is_a2a, is_equalxy, is_notest, ah_strategy, max_loop, tuning_mode, is_W0, extrapolation_window);
#else
    printf("Nx %d Ny %d Nz %d p %d p1 %d r %d a %d m %d o %d b %d e %d s %d l %d O %d Q %d A %d\n",
      Nx, Ny, Nz, p, p1, reps, fft_alg, fftw_tuning_level, is_oned, is_a2a, is_equalxy, ah_strategy, max_loop, tuning_mode, is_W0, extrapolation_window);
#endif
    if (fft_alg == 0) {
      printf("@ INPUT "); print_params(custom_params->v);
    }
  }

#ifdef DECOMP
  if (fft_alg == 3) {
    int Nz_new = (is_r2c)?Nz/2+1:Nz;
    int p2 = p / p1;
    if (Nx % p1 != 0 || Ny % p1 != 0 || Ny % p2 != 0 || Nz_new % p2 != 0) {
      if (!rank) {
        printf("Bad Arguments: Nx Ny Nz not divisible\n");
        printf("t_min ");
        printf("%.5f\n", t_min); /* 999999999.0 */
      }
      goto finish;
    }
  }
#endif

  /* allocate memory for input data */
  int size = -1;
  if (fft_alg == 0 && max_loop > 0) {
    int pi1 = -1, max_pi1 = -1;
    int Nz_new = (is_r2c)?Nz/2+1:Nz;
    size = -1;
    /*for (pi1 = 1; pi1 <= int(sqrt(p)); pi1++) { */
    int p_u = min( min(Nx,Ny), p );
    int p_l = max( max(p/Nz_new,p/Ny), 1 );
    for (pi1 = p_l; pi1 <= p_u; pi1++) {
      if (p % pi1 != 0) continue;
      int pi2 = p / pi1;
      int M1 = (Nx+pi1-1)/pi1;
      int M2 = (Ny+pi2-1)/pi2;
      int M3 = (is_r2c)?(Nz/2+1+pi2-1)/pi2:(Nz+pi2-1)/pi2;
      int M4 = (Ny+pi1-1)/pi1;
      //M1M2M3pi2 vs M1M3M4pi1
      int curr_size = (M2*pi2 > M4*pi1)? M1*M2*M3*pi2 : M1*M3*M4*pi1;
      if (curr_size > size) { size = curr_size; max_pi1 = pi1; }
    }
    if (!rank) printf("p1 for max input size %d\n", max_pi1);
  } else {
    if (p1 == -1) {
      p1 = p;
      if (!rank) printf("set p1 = %d\n", p1);
    }
    int p2 = p / p1;
    int M1 = (Nx+p1-1)/p1;
    int M2 = (Ny+p2-1)/p2;
    int M3 = (is_r2c)?(Nz/2+1+p2-1)/p2:(Nz+p2-1)/p2;
    int M4 = (Ny+p1-1)/p1;
    //M1M2M3p2 vs M1M3M4p1
    size = (M2*p2 > M4*p1)? M1*M2*M3*p2 : M1*M3*M4*p1;
  }
  if (!rank) printf("allocate memory for total # elements %d\n", size);
  //in = (double*)calloc(size * 2, sizeof(double));
  out = (double*)calloc(size * 2, sizeof(double));
  //out = (double*)fftw_malloc(size * 2 * sizeof(double));

  /* initialize */
  struct _offt_plan *po = NULL;
  MPI_Barrier(MPI_COMM_WORLD);
  t = 0.0;
  t -= MPI_Wtime();
  if (fft_alg == 0) {
#ifdef NOTEST
    po = offt_3d_init(Nx, Ny, Nz, out, out, is_r2c, fftw_flag,
      is_oned, is_a2a, is_equalxy, is_notest, ah_strategy, max_loop, tuning_mode, is_W0, extrapolation_window, custom_params);
#else
    po = offt_3d_init(Nx, Ny, Nz, out, out, is_r2c, fftw_flag,
      is_oned, is_a2a, is_equalxy, ah_strategy, max_loop, tuning_mode, is_W0, extrapolation_window, custom_params);
#endif
    p1 = po->params->v[_P1_];
    if (!rank) { printf("@ FINAL "); print_params(po->params->v); }
  } else if (fft_alg == 1) {
    /* p == p1 for FFTW */
    fftw_mpi_init();
    if (is_r2c) {
      mpi_p3d = fftw_mpi_plan_dft_r2c_3d(Nx, Ny, Nz,
        (double*)out, (fftw_complex*)out,
        MPI_COMM_WORLD,
        fftw_flag | FFTW_MPI_TRANSPOSED_OUT);
    } else {
      mpi_p3d = fftw_mpi_plan_dft_3d(Nx, Ny, Nz,
        (fftw_complex*)out, (fftw_complex*)out,
        MPI_COMM_WORLD, FFTW_FORWARD,
        fftw_flag | FFTW_MPI_TRANSPOSED_OUT);
    }
    if (!rank) {
      printf("==fftw\n");
      fftw_print_plan(mpi_p3d);
      printf("\n");
    }
#ifdef P3DFFT
  } else if (fft_alg == 2) {
    /* x and z are switched in p3dfft. */
    int dims[2] = {p / p1, p1};
    int memsize[3];
    p3dfft_setup(dims, Nz, Ny, Nx, 1, memsize);
#endif
#ifdef DECOMP
  } else if (fft_alg == 3) {
    decomp_2d_init(Nz, Ny, Nx, p / p1, p1, NULL);
    fft_init_arg(1); /* 1: start with x-pencil 3: start with z-pencil */
#endif
  } /* end of if fft_alg */
  t += MPI_Wtime();
  //MPI_Reduce(&t, &reduce_t, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  //MPI_Bcast(&reduce_t, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  //t = reduce_t;
  if (!rank) {
    printf("t_init ");
    if (fft_alg == 0) {
      po->t[INIT_ALL] = t;
      printf("%.5f %.5f %.5f %.5f\n",
        po->t_init[INIT_ALL], po->t_init[INIT_FFTW],
        po->t_init[INIT_AH], po->t_init[INIT_BUFFER]);
    } else 
      printf("%.5f\n", t);
  }

  /* computation */
  int r;
  for (r = 0; r < reps; r++) {
    double t_old = t;
    init(fft_alg, Nx, Ny, Nz, p1, p / p1, rank, is_r2c, out, out, po);
    MPI_Barrier(MPI_COMM_WORLD);
    if (r >= 0) t -= MPI_Wtime();
    if (fft_alg == 0) {
      /* input layout: x-y-z */
      /* x-stride: max((M2*p2)*M3, (M4*p1)*M3) */
      /* y-stride: M3*p2 */
      /* z-stride: 1 */
      offt_3d_execute(po, out, out, 0);
    } else if (fft_alg == 1) {
      fftw_execute(mpi_p3d); 
#ifdef P3DFFT
    } else if (fft_alg == 2) {
      unsigned char op_f[4] = "fft";
      p3dfft_ftran_r2c(out, out, op_f);
#endif
#ifdef DECOMP
    } else if (fft_alg == 3) {
      int dir = -1; /* -1:forward 1:backward */
      fft_3d_c2c(out, out, dir);
#endif
    }
    if (r >= 0) t += MPI_Wtime();
    //MPI_Reduce(&t, &reduce_t, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    //MPI_Bcast(&reduce_t, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    //t = reduce_t;
    double t_curr = t - t_old;
    if (!rank) {
      printf("t_%d ", r);
      if (fft_alg == 0) {
        po->t[ALL] = t_curr;
        offt_print_time(po->t);
      } else 
        printf("%.5f\n", t_curr);
    }
    if (t_curr < t_min) {
      t_min = t_curr;
      if (fft_alg == 0) {
        memcpy(offt_t_min, po->t, sizeof(double) * GES);
      }
    }
  }

  /* finalize */
  double t_old = t;
  MPI_Barrier(MPI_COMM_WORLD);
  t -= MPI_Wtime();
  if (fft_alg == 0) {
    offt_3d_fin(po);
  } else if (fft_alg == 1) {
    fftw_destroy_plan(mpi_p3d);
    fftw_mpi_cleanup();
#ifdef P3DFFT
  } else if (fft_alg == 2) {
    p3dfft_clean();
#endif
#ifdef DECOMP
  } else if (fft_alg == 3) {
    decomp_2d_fft_finalize();
#endif
  }
  t += MPI_Wtime();
  //MPI_Reduce(&t, &reduce_t, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  //MPI_Bcast(&reduce_t, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  //t = reduce_t;
  if (!rank) printf("t_fin %.5f\n", t-t_old);

  /* print total time */
  if (!rank) printf("t_all %.5f\n", t);
  if (!rank) {
    printf("t_min ");
    if (fft_alg == 0) {
      offt_print_time(offt_t_min);
    } else {
      printf("%.5f\n", t_min);
    }
  }

  /* print computation result */
  if (verbose && rank == 0)
  {
    int p2 = p / p1;
    int x = 0;
    int y = 0;
    int z = 0;
    int MM1 = Nx/p1;
    int MM3 = (is_r2c)?(Nz/2+1)/p2:Nz/p2;
    int MM4 = Ny/p1;
#if 1
    int xEnd = (1 > Nx)? Nx: 1;
    int yEnd = (1 > MM4)? MM4: 1;
    int zEnd = (4 > MM3)? MM3: 4;
#else
    int xEnd = Nx;
    int yEnd = Ny;
    int zEnd = 2;
#endif

    printf("p1 %d p2 %d MM3 %d MM4 %d\n", p1, p2, MM3, MM4);
    for (x = 0; x < xEnd; x++)
      for (y = 0; y < yEnd; y++)
        for (z = 0; z < zEnd; z++) {
          double a = 0.0, b = 0.0;
          if (fft_alg == 0) {
            a = out[0 + 2*(x*po->comm->ostride[0]+y*po->comm->ostride[1]+z*po->comm->ostride[2])];
            b = out[1 + 2*(x*po->comm->ostride[0]+y*po->comm->ostride[1]+z*po->comm->ostride[2])];
          } else if (fft_alg == 1) {
            /* output layout: y-x-z */
            if (is_r2c) {
              a = out[0 + 2*(z+(Nz/2+1)*(x+Nx*y))];
              b = out[1 + 2*(z+(Nz/2+1)*(x+Nx*y))];
            } else {
              a = out[0 + 2*(z+Nz*(x+Nx*y))];
              b = out[1 + 2*(z+Nz*(x+Nx*y))];
            }
#ifdef P3DFFT
          } else if (fft_alg == 2) {
            /* output layout: z-y-x */
            a = out[0 + 2*(x+Nx*(y+MM4*z))]; // TODO M4 should be m4 for the last process
            b = out[1 + 2*(x+Nx*(y+MM4*z))];
#endif
#ifdef DECOMP
          } else if (fft_alg == 3) {
            /* output layout: z-y-x */
            a = out[0 + 2*(z+MM3*(y+MM1*x))]; // TODO M4 should be m4 for the last process
            b = out[1 + 2*(z+MM3*(y+MM1*x))];
#endif
          }
          printf("p %d: %d %d %d: %.5f %.5f\n", rank, x, y, z, a, b);
        } /* end of for z */
  } /* end of if verbose */
  //if (!rank) printf("%.5f\n", t_min);

  /* print computation result */
  //free(in);
  free(out);
  //fftw_free(out);
  free(custom_params);
  out = NULL;
finish:
  custom_params = NULL;
  MPI_Finalize();
}
