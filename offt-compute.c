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

#include <string.h>
#include <stdlib.h>
/* stupid MPICH2 hack macro */
//#define MPICH_IGNORE_CXX_SEEK 
#include <mpi.h>
#ifdef SHSONG_MPICH
#else
#include <nbc.h>
#endif
#include <math.h>
#include <unistd.h>
#include "offt-internal.h"

/* **********************************************************
 @ basic functions
   ********************************************************** */
static int inv_log(int n) {
  return (n == LOG0)? 0: (1 << n);
}

static int floor_log(int n) {
  // to support F_FFT = 0
  if (n == 0) return LOG0;
  // assume n > 0
  int count = -1;
  while (n > 0) {
    count++;
    n = n >> 1;
  }
  //printf("n %d -> floor_log_n %d\n", oldn, count);
  return count;
}

/* **********************************************************
 @ setup "struct _offt_comm"
   ********************************************************** */
struct _offt_comm* offt_comm_malloc(struct _offt_plan* po) {
  struct _offt_comm* comm = (struct _offt_comm*)malloc(sizeof(struct _offt_comm));
  /* default input parameters */
  int Nx = po->Nx;
  int Ny = po->Ny;
  int Nz = po->Nz;
  int Nz_new = (po->is_r2c)?(Nz/2+1):Nz;
  int p1 = comm->p1 = po->params->v[_P1_];
  int p = po->p, p2;
  int rank = po->rank;
  comm->p2 = p2 = p / p1;
  
#ifdef ROTATE_RANKORDER
  /* rank => x-coord=rank%p1, y-coord=rank/p1 */
  int rank_x = rank%p1;
  int rank_y = rank/p1;
#else
  /* rank => x-coord=rank/p2, y-coord=rank%p2 */
  int rank_x = rank/p2;
  int rank_y = rank%p2;
#endif
  /* set comm1 and comm2
     comm1 : i*p2,...,(i+1)*p2-1
     comm2 : i+0*p2,i+1*p2,...
            +----+----+----+----+
  ranks2[2] |  2 |  5 |  8 | 11 |
            +----+----+----+----+
  ranks2[1] |  1 |  4 |  7 | 10 |
            +----+----+----+----+
  ranks2[0] |  0 |  3 |  6 |  9 |
            +----+----+----+----+
  */
  int ranks1[p1][p2];
  int ranks2[p2][p1];
  int i, j;
  //int *ranks2 = (int*)malloc(p2 * p1 * sizeof(int));
  for (i = 0; i < p1; i++) {
    for (j = 0; j < p2; j++) {
#ifdef ROTATE_RANKORDER
      ranks1[i][j] = i + j * p1;
      ranks2[j][i] = i + j * p1;
#else
      ranks1[i][j] = j + i * p2;
      ranks2[j][i] = j + i * p2;
#endif
      //if (!comm->rank) {
      //  printf("ranks1[%d][%d] = %d\n", i, j, ranks1[i][j]);
      //}
    }
  }

  comm->group1 = (MPI_Group*)malloc(sizeof(MPI_Group));
  comm->group2 = (MPI_Group*)malloc(sizeof(MPI_Group));
  MPI_Group orig_group;
  MPI_Comm_group(MPI_COMM_WORLD, &orig_group);
#ifdef ROTATE_RANKORDER
  // rank => x-coord=rank%p1, y-coord=rank/p1
  MPI_Group_incl(orig_group, p2, ranks1[rank_x], comm->group1);
  MPI_Group_incl(orig_group, p1, ranks2[rank_y], comm->group2);
#else
  // rank => x-coord=rank/p2, y-coord=rank%p2
  MPI_Group_incl(orig_group, p2, ranks1[rank_x], comm->group1);
  MPI_Group_incl(orig_group, p1, ranks2[rank_y], comm->group2);
#endif

  comm->comm1 = (MPI_Comm*)malloc(sizeof(MPI_Comm));
  comm->comm2 = (MPI_Comm*)malloc(sizeof(MPI_Comm));
  MPI_Comm_create(MPI_COMM_WORLD, *(comm->group1), comm->comm1);
  MPI_Comm_create(MPI_COMM_WORLD, *(comm->group2), comm->comm2);

#ifdef A2AV
  comm->M1 = (Nx + p1 - 1) / p1;
  comm->M2 = (Ny + p2 - 1) / p2;
  comm->M3 = (Nz_new + p2 - 1) / p2;
  comm->M4 = (Ny + p1 - 1) / p1;
  comm->F1 = Nx / p1;
  comm->F2 = Ny / p2;
  comm->F3 = Nz_new / p2;
  comm->F4 = Ny / p1;
  comm->b1 = Nx % p1;
  comm->b2 = Ny % p2;
  comm->b3 = Nz_new % p2;
  comm->b4 = Ny % p1;
  /* rank => x-coord=rank/p2, y-coord=rank%p2 */
  comm->m1 = (rank_x < p1 - comm->b1)? Nx / p1: Nx / p1 + 1;
  comm->m2 = (rank_y < p2 - comm->b2)? Ny / p2: Ny / p2 + 1;
  comm->m3 = (rank_y < p2 - comm->b3)? Nz_new / p2: Nz_new / p2 + 1;
  comm->m4 = (rank_x < p1 - comm->b4)? Ny / p1: Ny / p1 + 1;
#else
  /* calculate distribution parameters */
  // C2C: M1=M4, M2=M3, p1=p2 => M1=M2=M3=M4
  comm->M1 = (Nx + p1 - 1) / p1;
  comm->m1 = comm->M1;
  comm->M2 = (Ny + p2 - 1) / p2;
  comm->m2 = comm->M2;
  comm->M3 = (Nz_new + p2 - 1) / p2;
  comm->m3 = comm->M3;
  comm->M4 = (Ny + p1 - 1) / p1;
  comm->m4 = comm->M4;

  /* adjust m1, m2, m3, m4 for processes at edge */
  for (i = Nx/comm->M1; i < p1; i++)
    for (j = 0; j < p2; j++)
      if (ranks1[i][j] == rank) {
        /*comm->m1 = (Nx % comm->M1 == 0)? comm->M1: (Nx % comm->M1);*/
        if (Nx % comm->M1 == 0) {
          if (comm->M1 * p1 > Nx) {
            comm->m1 = 0;
          } else if (comm->M1 * p1 == Nx) {
            /* should not be reached */
          } else if (comm->M1 * p1 < Nx) {
            /* impossible */
          }
        } else {
          if (comm->M1 * p1 > Nx) {
            comm->m1 = (i == Nx/comm->M1)? Nx % comm->M1: 0;
          } else {
            /* impossible */
          }
        }
      }
  for (i = Ny/comm->M4; i < p1; i++)
    for (j = 0; j < p2; j++)
      if (ranks1[i][j] == rank) {
        /*comm->m4 = (Ny % comm->M4 == 0)? comm->M4: (Ny % comm->M4);*/
        if (Ny % comm->M4 == 0) {
          if (comm->M4 * p1 > Ny) {
            comm->m4 = 0;
          } else if (comm->M4 * p1 == Ny) {
            /* should not be reached */
          } else if (comm->M4 * p1 < Ny) {
            /* impossible */
          }
        } else {
          if (comm->M4 * p1 > Ny) {
            comm->m4 = (i == Ny/comm->M4)? Ny % comm->M4: 0;
          } else {
            /* impossible */
          }
        }
      }
  for (i = Ny/comm->M2; i < p2; i++)
    for (j = 0; j < p1; j++)
      if (ranks2[i][j] == rank) {
        /*comm->m2 = (Ny % comm->M2 == 0)? comm->M2: (Ny % comm->M2);*/
        if (Ny % comm->M2 == 0) {
          if (comm->M2 * p2 > Ny) {
            comm->m2 = 0;
          } else if (comm->M2 * p2 == Ny) {
            /* should not be reached */
          } else if (comm->M2 * p2 < Ny) {
            /* impossible */
          }
        } else {
          if (comm->M2 * p2 > Ny) {
            comm->m2 = (i == Ny/comm->M2)? Ny % comm->M2: 0;
          } else {
            /* impossible */
          }
        }
      }
  for (i = Nz_new/comm->M3; i < p2; i++)
    for (j = 0; j < p1; j++)
      if (ranks2[i][j] == rank) {
        /*comm->m3 = (Nz_new % comm->M3 == 0)? comm->M3: (Nz_new % comm->M3);*/
        if (Nz_new % comm->M3 == 0) {
          if (comm->M3 * p2 > Nz_new) {
            comm->m3 = 0;
          } else if (comm->M3 * p2 == Nz_new) {
            /* should not be reached */
          } else if (comm->M3 * p2 < Nz_new) {
            /* impossible */
          }
        } else {
          if (comm->M3 * p2 > Nz_new) {
            comm->m3 = (i == Nz_new/comm->M3)? Nz_new % comm->M3: 0;
          } else {
            /* impossible */
          }
        }
      }
#endif // A2AV

#if 0
  if (!rank) {
    printf("M1 %d M2 %d M3 %d M4 %d m1 %d m2 %d m3 %d m4 %d\n", comm->M1, comm->M2, comm->M3, comm->M4, comm->m1, comm->m2, comm->m3, comm->m4);
  }
#endif
#ifdef A2AV
  comm->istart[0] = (rank_x < p1 - comm->b1)? rank_x*comm->F1: (p1-comm->b1)*comm->F1+(rank_x-(p1-comm->b1))*(comm->F1+1);
  comm->istart[1] = (rank_y < p2 - comm->b2)? rank_y*comm->F2: (p2-comm->b2)*comm->F2+(rank_y-(p2-comm->b2))*(comm->F2+1);
  comm->istart[2] = 0;
  comm->isize[0] = comm->m1;
  comm->isize[1] = comm->m2;
  comm->isize[2] = Nz;
#else
  comm->istart[0] = rank_x*comm->M1;
  comm->istart[1] = rank_y*comm->M2;
  comm->istart[2] = 0;
  comm->isize[0] = comm->m1;
  comm->isize[1] = comm->m2;
  comm->isize[2] = Nz;
#endif
  comm->istride[0] = (comm->M2*comm->p2 > comm->M4*comm->p1)?
    comm->M2*comm->M3*comm->p2: comm->M4*comm->p1*comm->M3;
  comm->istride[1] = comm->M3*comm->p2;
  comm->istride[2] = 1;

  //if (!rank) { printf("istride %d %d %d\n", comm->istride[0], comm->istride[1], comm->istride[2]);}
  comm->ostart[0] = 0;
#ifdef A2AV
  comm->ostart[1] = (rank_x < p1-comm->b4)?
    rank_x*comm->F4: (p1-comm->b4)*comm->F4+(rank_x-(p1-comm->b4))*(comm->F4+1);
  comm->ostart[2] = (rank_y < p2-comm->b3)?
    rank_y*comm->F3: (p2-comm->b3)*comm->F3+(rank_y-(p2-comm->b3))*(comm->F3+1);
  comm->osize[0] = Nx;
  comm->osize[1] = comm->m4;
  comm->osize[2] = comm->m3;
#else
  comm->ostart[1] = rank_x*comm->M4;
  comm->ostart[2] = rank_y*comm->M3;
  comm->osize[0] = Nx;
  comm->osize[1] = comm->m4;
  comm->osize[2] = comm->m3;
#endif
#ifdef STRIDE
  if (po->params->v[_S_]) {
    /* x-y-z */
    comm->ostride[0] = comm->M3*comm->M4;
    comm->ostride[1] = comm->M3;
    comm->ostride[2] = 1;
  } else {
    if (po->is_equalxy && comm->M1 == comm->M4) {
      /* y-z-x */
      comm->ostride[0] = 1;
      comm->ostride[1] = comm->M1*comm->p1 * comm->M3;
      comm->ostride[2] = comm->M1*comm->p1;
    } else {
      /* z-y-x */
      comm->ostride[0] = 1;
      comm->ostride[1] = comm->M1*comm->p1;
      comm->ostride[2] = comm->M1*comm->p1 * comm->M4;
    }
  }
#else
  if (po->is_equalxy && comm->M1 == comm->M4) {
    /* y-z-x */
    comm->ostride[0] = 1;
    comm->ostride[1] = comm->M1*comm->p1 * comm->M3;
    comm->ostride[2] = comm->M1*comm->p1;
  } else {
    /* z-y-x */
    comm->ostride[0] = 1;
    comm->ostride[1] = comm->M1*comm->p1;
    comm->ostride[2] = comm->M1*comm->p1 * comm->M4;
  }
#endif
  return comm;
}

void offt_comm_free(struct _offt_comm *comm) {
  free(comm->group1);
  free(comm->group2);
  free(comm->comm1);
  free(comm->comm2);
  free(comm);
}

#ifdef P1D_LIST
/* **********************************************************
 @ setup po->pt_transpose
   ********************************************************** */
void setup_p1d(struct _offt_plan* po, double *out) {
  po->t_init[INIT_FFTW] -= MPI_Wtime();

  /* p1d_z */
  if (!po->p1d_z) {
    if (po->is_r2c) {
      po->p1d_z = fftw_plan_dft_r2c_1d(po->Nz,
        (double*)out, (fftw_complex*)out, po->fftw_flag);
    } else {
      po->p1d_z = fftw_plan_dft_1d(po->Nz,
        (fftw_complex*)out, (fftw_complex*)out,
        FFTW_FORWARD, po->fftw_flag);
    }
#if 0
    if (!po->rank) {
      printf("==p1d_z\n");
      fftw_print_plan(po->p1d_z);
      printf("\n");
    }
#endif
  }

  /* p1d_y and p1d_x */
#ifdef STRIDE
  if (po->params->v[_S_]) {
    int c = 0; /* index number of p1d_x_s_list element */
    struct _offt_comm* comm = po->comm;

    if (po->max_loop > 0) {
      int pi1;
      /* memory assign */
      if (po->p1d_x_s_list == NULL) {
        for (pi1 = 1; pi1 <= po->p; pi1++) {
          if (po->p % pi1 != 0) continue;
          c++;
        }
        po->p1d_xy_s_list_size = c;
        po->p1d_x_s_list = (fftw_plan*)malloc(po->p1d_xy_s_list_size * sizeof(fftw_plan));
        po->p1d_y_s_list = (fftw_plan*)malloc(po->p1d_xy_s_list_size * sizeof(fftw_plan));
        int i;
        for (i = 0; i < po->p1d_xy_s_list_size; i++) {
          po->p1d_x_s_list[i] = NULL;
          po->p1d_y_s_list[i] = NULL;
        }
      }

      /* check if plan already exists */
      c = 0; /* index */
      for (pi1 = 1; pi1 <= po->p; pi1++) {
        if (po->p % pi1 != 0) continue;
        if (pi1 == po->comm->p1) break;
        c++;
      }
      if (po->p1d_x_s_list[c]) {
        po->p1d_y = po->p1d_y_s_list[c];
        po->p1d_x = po->p1d_x_s_list[c];
        goto finish_setup_p1d;
      }
    }

    {
      int n_y[1] = { po->Ny };
      int n_x[1] = { po->Nx };
      int howmany_y = 1;
      int stride_y = 1;
      int dist_y = 0;
      int howmany_x = 1;
      int stride_x = 1;
      int dist_x = 0;
      stride_y = comm->M3; /* xyz */
      stride_x = comm->M3 * comm->M4; /* xyz */
      if (po->is_oned && comm->p1 == 1) {
        howmany_x = comm->M3 * comm->M4;
        stride_x = comm->M3 * comm->M4; /* xyz */
        dist_x = 1;
      }
  #if 0
      stride_y = comm->M3; /* xyz */
      stride_x = comm->M3; /* yxz */
      if (po->is_oned && comm->p1 == 1) {
        //howmany_x = comm->M3;
        //stride_x = comm->M3; /* yxz */
        howmany_x = comm->M3 * comm->M4;
        stride_x = comm->M3 * comm->M4; /* xyz */
        dist_x = 1;
      }
  #endif
      po->p1d_y = fftw_plan_many_dft(
        1, n_y, howmany_y, /* rank, n, howmany, */
        (fftw_complex*)out, n_y, stride_y, dist_y,
        (fftw_complex*)out, n_y, stride_y, dist_y,
        FFTW_FORWARD, po->fftw_flag);
      po->p1d_x = fftw_plan_many_dft(
        1, n_x, howmany_x, /* rank, n, howmany, */
        (fftw_complex*)out, n_x, stride_x, dist_x,
        (fftw_complex*)out, n_x, stride_x, dist_x,
        FFTW_FORWARD, po->fftw_flag);
    }
    if (po->max_loop > 0) {
      po->p1d_y_s_list[c] = po->p1d_y;
      po->p1d_x_s_list[c] = po->p1d_x;
    }
#if 0
    if (!po->rank) {
      printf("==p1d_y\n");
      fftw_print_plan(po->p1d_y);
      printf("\n");
      printf("==p1d_x\n");
      fftw_print_plan(po->p1d_x);
      printf("\n");
    }
#endif
  } else {
    if (!po->p1d_y_t) {
      po->p1d_y_t = fftw_plan_dft_1d(po->Ny,
        (fftw_complex*)out, (fftw_complex*)out,
        FFTW_FORWARD, po->fftw_flag);
      po->p1d_x_t = fftw_plan_dft_1d(po->Nx,
        (fftw_complex*)out, (fftw_complex*)out,
        FFTW_FORWARD, po->fftw_flag);
#if 0
      if (!po->rank) {
        printf("==p1d_y\n");
        fftw_print_plan(po->p1d_y_t);
        printf("\n");
        printf("==p1d_x\n");
        fftw_print_plan(po->p1d_x_t);
        printf("\n");
      }
#endif
    }
    po->p1d_y = po->p1d_y_t;
    po->p1d_x = po->p1d_x_t;
  }
#else // STRIDE
  if (!po->p1d_y) {
    po->p1d_y = fftw_plan_dft_1d(po->Ny,
      (fftw_complex*)out, (fftw_complex*)out,
      FFTW_FORWARD, po->fftw_flag);
    po->p1d_x = fftw_plan_dft_1d(po->Nx,
      (fftw_complex*)out, (fftw_complex*)out,
      FFTW_FORWARD, po->fftw_flag);
#if 0
    if (!po->rank) {
      printf("==p1d_y\n");
      fftw_print_plan(po->p1d_y);
      printf("\n");
      printf("==p1d_x\n");
      fftw_print_plan(po->p1d_x);
      printf("\n");
    }
#endif
  }
#endif // STRIDE

#ifdef STRIDE
finish_setup_p1d:
#endif
  po->t_init[INIT_FFTW] += MPI_Wtime();
  return;
}

void clear_p1d(struct _offt_plan* po) {
  if (po->p1d_z)
    fftw_destroy_plan(po->p1d_z);
  if (po->p1d_y_t)
    fftw_destroy_plan(po->p1d_y_t);
  if (po->p1d_x_t)
    fftw_destroy_plan(po->p1d_x_t);
  if (po->p1d_xy_s_list_size > 0) {
    int i;
    for (i = 0; i < po->p1d_xy_s_list_size; i++) {
      if (po->p1d_y_s_list[i])
        fftw_destroy_plan(po->p1d_y_s_list[i]);
      if (po->p1d_x_s_list[i])
        fftw_destroy_plan(po->p1d_x_s_list[i]);
    }
    free(po->p1d_y_s_list);
    free(po->p1d_x_s_list);
    po->p1d_y_s_list = NULL;
    po->p1d_x_s_list = NULL;
    po->p1d_xy_s_list_size = 0;
  }
  po->p1d_z = NULL;
  po->p1d_y = NULL;
  po->p1d_x = NULL;
  po->p1d_y_t = NULL;
  po->p1d_x_t = NULL;
}
#endif // end of P1D_LIST

/* **********************************************************
 @ setup po->pt_transpose
   ********************************************************** */
fftw_plan setup_transpose(struct _offt_plan* po, double *out) {
  po->t_init[INIT_FFTW] -= MPI_Wtime();
#ifdef TRANSPOSE_LIST
  int c = 0;

  if (po->max_loop > 0) {
    int pi1;
    /* memory assign */
    if (po->pt_transpose_list == NULL) {
      for (pi1 = 1; pi1 <= po->p; pi1++) {
        if (po->p % pi1 != 0) continue;
        c++;
      }
      po->pt_transpose_list_size = c;
      po->pt_transpose_list = (fftw_plan*)malloc(po->pt_transpose_list_size * sizeof(fftw_plan));
      int i;
      for (i = 0; i < po->pt_transpose_list_size; i++)
        po->pt_transpose_list[i] = NULL;
    }

    /* check if plan already exists */
    c = 0; /* index */
    for (pi1 = 1; pi1 <= po->p; pi1++) {
      if (po->p % pi1 != 0) continue;
      if (pi1 == po->comm->p1) break;
      c++;
    }
    if (po->pt_transpose_list[c]) {
      po->t_init[INIT_FFTW] += MPI_Wtime();
      return po->pt_transpose_list[c];
    }
  }
#endif

  struct _offt_comm* comm = po->comm;
  /* for fast transpose */
  fftw_iodim howmany_dims[3];
  /* howmany_dims[d].in: how many strides are needed for dim d in an input */
  /* howmany_dims[d].os: how many strides are needed for dim d in a new output */
  if (po->is_oned && comm->p1 == 1) {
    if (po->is_equalxy && comm->M1 == comm->M4) {
      /* xzy -> yzx: when Nx == Ny */
      howmany_dims[0].n = comm->M1; /* Nx */
      howmany_dims[0].is = comm->M4 * comm->M3;
      howmany_dims[0].os = 1;
      howmany_dims[1].n = comm->M3;
      howmany_dims[1].is = comm->M4;
      howmany_dims[1].os = comm->M1;
      howmany_dims[2].n = comm->M4; /* Ny */
      howmany_dims[2].is = 1;
      howmany_dims[2].os = comm->M1 * comm->M3;
    } else {
      /* xzy -> zyx */
      howmany_dims[0].n = comm->M1; /* Nx */
      howmany_dims[0].is = comm->M4 * comm->M3;
      howmany_dims[0].os = 1;
      howmany_dims[1].n = comm->M3;
      howmany_dims[1].is = comm->M4;
      howmany_dims[1].os = comm->M1 * comm->M4;
      howmany_dims[2].n = comm->M4; /* Ny */
      howmany_dims[2].is = 1;
      howmany_dims[2].os = comm->M1;
    }
  } else if (po->is_oned && comm->p1 == po->p) {
    if (po->is_equalxy && comm->M1 == comm->M4) {
      /* xyz -> xzy: when Nx == Ny */
      howmany_dims[0].n = comm->M1; /* x */
      howmany_dims[0].is = (comm->M4 * comm->p1) * comm->M3;
      howmany_dims[0].os = (comm->M4 * comm->p1) * comm->M3;
      howmany_dims[1].n = comm->M4 * comm->p1;
      /* y: M2*p2 <= M4*p1 => x-stride = M4*p1*M3 => M4*p1 = new y-stride */
      howmany_dims[1].is = comm->M3;
      howmany_dims[1].os = 1;
      howmany_dims[2].n = comm->M3; /* z: Nz */
      howmany_dims[2].is = 1;
      howmany_dims[2].os = comm->M4 * comm->p1;
    } else {
      /* xyz -> zxy */
      howmany_dims[0].n = comm->M1; /* x */
      howmany_dims[0].is = (comm->M4 * comm->p1) * comm->M3;
      howmany_dims[0].os = comm->M4 * comm->p1;
      howmany_dims[1].n = comm->M4 * comm->p1;
      /* y: M2*p2 <= M4*p1 => x-stride = M4*p1*M3 => M4*p1 = new y-stride */
      howmany_dims[1].is = comm->M3;
      howmany_dims[1].os = 1;
      howmany_dims[2].n = comm->M3; /* z: Nz */
      howmany_dims[2].is = 1;
      howmany_dims[2].os = (comm->M4 * comm->p1) * comm->M1;
    }
  } else {
    if (po->is_equalxy && comm->M1 == comm->M4) {
      /* no transpose required */
      howmany_dims[0].n = 1;
      howmany_dims[0].is = 1;
      howmany_dims[0].os = 1;
      howmany_dims[1].n = 1;
      howmany_dims[1].is = 1;
      howmany_dims[1].os = 1;
      howmany_dims[2].n = 1;
      howmany_dims[2].is = 1;
      howmany_dims[2].os = 1;
    } else {
      /* xzy -> zxy for tiling, in-place, Nx!=Ny */
      howmany_dims[0].n = comm->M1; /* x */
      howmany_dims[0].is = (comm->M4*comm->p1) * comm->M3;
      howmany_dims[0].os = (comm->M4*comm->p1);
      howmany_dims[1].n = comm->M3; /* z */
      howmany_dims[1].is = (comm->M4*comm->p1);
      howmany_dims[1].os = (comm->M4*comm->p1) * comm->M1;
      howmany_dims[2].n = (comm->M4*comm->p1); /* y */
      howmany_dims[2].is = 1;
      howmany_dims[2].os = 1;
    }
  }
  fftw_plan fp = fftw_plan_guru_dft(0, NULL, 3, howmany_dims,
      (fftw_complex*)out, (fftw_complex*)out,
      -1, po->fftw_flag);
#ifdef TRANSPOSE_LIST
  if (po->max_loop > 0)
    po->pt_transpose_list[c] = fp;
#endif
  po->t_init[INIT_FFTW] += MPI_Wtime();
#if 0
  if (!po->rank) {
    printf("==pt_transpose\n");
    fftw_print_plan(fp);
    printf("\n");
  }
#endif
  return fp;
}

#ifdef TRANSPOSE_LIST
void clear_transpose(struct _offt_plan* po) {
  int i;
  for (i = 0; i < po->pt_transpose_list_size; i++) {
    if (po->pt_transpose_list[i])
      fftw_destroy_plan(po->pt_transpose_list[i]);
  }
  free(po->pt_transpose_list);
  po->pt_transpose_list = NULL;
  po->pt_transpose_list_size = 0;
  po->pt_transpose = NULL;
}
#endif

/* **********************************************************
 @ setup "struct _offt_buffer"
   ********************************************************** */
struct _offt_buffer {
  double *a2as;
  double *a2ar;
  long size; /* # double elements */
#ifdef SHSONG_MPICH
  MPI_Request handle;
#else
  NBC_Handle handle;
#endif
};

#ifdef BUFMALLOC
void set_buffer_chunk(struct _offt_plan *po, int is_tuning) {
  po->t_init[INIT_BUFFER] -= MPI_Wtime();
  size_t size = 0;
  if (is_tuning) {
    int Nz_new = (po->is_r2c)?(po->Nz/2+1):po->Nz;
    /* chunk size >= max((T1)*(W1+1)*2, (T2)*(W2+1)) */
    /* buffer size overhead <=256MB (16M elems) * (2 (a2as+a2ar) + 1 (additional)) */
    /* 2: a2as+a2ar, 2: complex number, 4: additional */
    size_t size1 = BUFFER_SIZE_LIMIT;
    size_t size2 = 2 * 2 * ((size_t)po->Nx * po->Ny / po->p) * Nz_new * 4;
    size = size1 <= size2? size1: size2;
  } else {
    /* 2: complex number */ /* 2: a2as+a2ar */
    size_t size1 = 2 * 2 * po->params->v[_T1_] * po->comm->M2 * (po->comm->M3 * po->comm->p2) * (po->params->v[_W1_]+1);
    size_t size2 = 2 * 2 * po->comm->M1 * (po->comm->M4 * po->comm->p1) * po->params->v[_T2_] * (po->params->v[_W2_]+1);
    size = size1 > size2? size1: size2;
  }
#if 1
  if (MPI_SUCCESS != MPI_Alloc_mem(size * sizeof(double), MPI_INFO_NULL, &(po->buffer_chunk))) {
    printf("Error allocating memory");
  }
#else
  po->buffer_chunk = (void*)malloc(size * sizeof(double));
#endif
  //if (!po->rank) printf("allocate buffer memory %zd elements at %p\n", size, po->buffer_chunk);
  po->t_init[INIT_BUFFER] += MPI_Wtime();
}

void set_buffer(struct _offt_plan *po) {
  po->t_init[INIT_BUFFER] -= MPI_Wtime();

  int W1 = po->params->v[_W1_];
  int W2 = po->params->v[_W2_];
  struct _offt_buffer **buffers1 = NULL;
  struct _offt_buffer **buffers2 = NULL;
  buffers1 = (struct _offt_buffer**)malloc((W1 + 1) * sizeof(struct _offt_buffer*));
  buffers2 = (struct _offt_buffer**)malloc((W2 + 1) * sizeof(struct _offt_buffer*));

  int size1 = 2 * po->params->v[_T1_] * po->comm->M2 * (po->comm->M3 * po->comm->p2);
  int size2 = 2 * po->comm->M1 * (po->comm->M4 * po->comm->p1) * po->params->v[_T2_];
  int i;
  double* chunk = (double*)po->buffer_chunk;
  for (i = 0; i < W1 + 1; i++) { 
    struct _offt_buffer* buf = (struct _offt_buffer*)malloc(sizeof(struct _offt_buffer));
    buf->size = size1;
    buf->a2as = chunk; chunk += buf->size;
    buf->a2ar = chunk; chunk += buf->size;
    //if (!po->rank) printf("W1 %d a2as %p a2ar %p size1 %zd\n", i, buf->a2as, buf->a2ar, buf->size); // TODEL
    buffers1[i] = buf;
  }
  chunk = (double*)po->buffer_chunk;
  for (i = 0; i < W2 + 1; i++) { 
    struct _offt_buffer* buf = (struct _offt_buffer*)malloc(sizeof(struct _offt_buffer));
    buf->size = size2;
    buf->a2as = chunk; chunk += buf->size;
    buf->a2ar = chunk; chunk += buf->size;
    //if (!po->rank) printf("W2 %d a2as %p a2ar %p size2 %zd\n", i, buf->a2as, buf->a2ar, buf->size); // TODEL
    buffers2[i] = buf;
  }
  po->buffers1 = (void*)buffers1;
  po->buffers2 = (void*)buffers2;
  po->t_init[INIT_BUFFER] += MPI_Wtime();
}

void clear_buffer(struct _offt_plan *po) {
  struct _offt_buffer **buffers1 = (struct _offt_buffer**)(po->buffers1);
  struct _offt_buffer **buffers2 = (struct _offt_buffer**)(po->buffers2);
  int i;
  for (i = 0; i < po->params->v[_W1_] + 1; i++) { 
    free(buffers1[i]);
  }
  for (i = 0; i < po->params->v[_W2_] + 1; i++) { 
    free(buffers2[i]);
  }
  free(buffers1);
  free(buffers2);
  po->buffers1 = NULL;
  po->buffers2 = NULL;
}
#else
struct _offt_buffer* buffer_malloc(struct _offt_comm * comm, int size)
{
  struct _offt_buffer* buf = (struct _offt_buffer*)malloc(sizeof(struct _offt_buffer));
  buf->size = size;
#if 1
  if(MPI_SUCCESS != MPI_Alloc_mem( buf->size*sizeof(double), MPI_INFO_NULL, &(buf->a2as))) {
    printf("Error allocating memory");
  }
  if(MPI_SUCCESS != MPI_Alloc_mem( buf->size*sizeof(double), MPI_INFO_NULL, &(buf->a2ar))) {
    printf("Error allocating memory");
  }
#else
  buf->a2as = (double*)malloc(buf->size * sizeof(double));
  buf->a2ar = (double*)malloc(buf->size * sizeof(double));
#endif
  //memset(buf->a2as, 0, buf->size*sizeof(double));
  //memset(buf->a2ar, 0, buf->size*sizeof(double));
  return buf;
}

void buffer_free(struct _offt_buffer* buf) {
#if 1
  MPI_Free_mem(buf->a2as);
  MPI_Free_mem(buf->a2ar);
#else
  free(buf->a2as);
  free(buf->a2ar);
#endif
  free(buf);
}

void set_buffer(struct _offt_plan *po) {
  po->t_init[INIT_BUFFER] -= MPI_Wtime();
  int size1 = 2 * po->params->v[_T1_] * po->comm->M2 * (po->comm->M3 * po->comm->p2);
  int size2 = 2 * po->comm->M1 * (po->comm->M4 * po->comm->p1) * po->params->v[_T2_];
  int W1 = po->params->v[_W1_];
  int W2 = po->params->v[_W2_];
  struct _offt_buffer **buffers1 = (struct _offt_buffer**)malloc((W1 + 1) * sizeof(struct _offt_buffer*));
  struct _offt_buffer **buffers2 = (struct _offt_buffer**)malloc((W2 + 1) * sizeof(struct _offt_buffer*));
  int i;
  for (i = 0; i < W1 + 1; i++) { 
    struct _offt_buffer* single_buffer = buffer_malloc(po->comm, size1);
    buffers1[i] = single_buffer;
  }
  for (i = 0; i < W2 + 1; i++) { 
    struct _offt_buffer* single_buffer = buffer_malloc(po->comm, size2);
    buffers2[i] = single_buffer;
  }
  po->buffers1 = (void*)buffers1;
  po->buffers2 = (void*)buffers2;
  po->t_init[INIT_BUFFER] += MPI_Wtime();
}

void clear_buffer(struct _offt_plan *po) {
  struct _offt_buffer **buffers1 = (struct _offt_buffer**)(po->buffers1);
  struct _offt_buffer **buffers2 = (struct _offt_buffer**)(po->buffers2);
  int i;
  for (i = 0; i < po->params->v[_W1_] + 1; i++) { 
    buffer_free(buffers1[i]);
  }
  for (i = 0; i < po->params->v[_W2_] + 1; i++) { 
    buffer_free(buffers2[i]);
  }
  free(buffers1);
  free(buffers2);
}
#endif

/* **********************************************************
 @ communication
   ********************************************************** */
#ifdef A2AV
static __inline__ void communicate_a2av(struct _offt_buffer *buffer, int is_blocking, MPI_Comm *pmpi_comm, int* sendcounts, int* sdispls, int* recvcounts, int* rdispls) {
  MPI_Comm mpi_comm = *pmpi_comm;
  if (is_blocking) {
    MPI_Alltoallv(buffer->a2as, sendcounts, sdispls, MPI_DOUBLE,
                  buffer->a2ar, recvcounts, rdispls, MPI_DOUBLE,
                  mpi_comm);
  } else {
#ifdef SHSONG_MPICH
#ifdef SHSONG_HOPPER
    MPI_Ialltoallv(buffer->a2as, sendcounts, sdispls, MPI_DOUBLE,
                   buffer->a2ar, recvcounts, rdispls, MPI_DOUBLE,
                   mpi_comm, &buffer->handle);
#else
    MPI_Ialltoallv(buffer->a2as, sendcounts, sdispls, MPI_DOUBLE,
                   buffer->a2ar, recvcounts, rdispls, MPI_DOUBLE,
                   mpi_comm, &buffer->handle);
#endif
#else
    NBC_Ialltoallv(buffer->a2as, sendcounts, sdispls, MPI_DOUBLE,
                   buffer->a2ar, recvcounts, rdispls, MPI_DOUBLE,
                   mpi_comm, &buffer->handle);
#endif
  }
}
#endif

static __inline__ void communicate_a2a(struct _offt_buffer *buffer, int is_blocking, MPI_Comm *pmpi_comm, int size) {
  MPI_Comm mpi_comm = *pmpi_comm;
  if (is_blocking) {
    MPI_Alltoall(buffer->a2as, size, MPI_DOUBLE, buffer->a2ar, size, MPI_DOUBLE,
                 mpi_comm);
  } else {
#ifdef SHSONG_MPICH
#ifdef SHSONG_HOPPER
    MPI_Ialltoall(buffer->a2as, size, MPI_DOUBLE, buffer->a2ar, size, MPI_DOUBLE,
                   mpi_comm, &buffer->handle);
#else
    MPI_Ialltoall(buffer->a2as, size, MPI_DOUBLE, buffer->a2ar, size, MPI_DOUBLE,
                  mpi_comm, &buffer->handle);
#endif
#else
    NBC_Ialltoall(buffer->a2as, size, MPI_DOUBLE, buffer->a2ar, size, MPI_DOUBLE,
                  mpi_comm, &buffer->handle);
#endif
  }
}
 
static __inline__ void communicate_wait(struct _offt_buffer *buffer) {
  MPI_Status status;
#ifdef SHSONG_MPICH
  MPI_Wait(&buffer->handle, &status);
#else
  NBC_Wait(&buffer->handle, &status);
#endif
}

static __inline__ void communicate_test(struct _offt_buffer *buffer) {
  int flag;
  MPI_Status status;
#ifdef SHSONG_MPICH
  MPI_Test(&buffer->handle, &flag, &status);
#else
  NBC_Test(&buffer->handle, &flag, &status);
#endif
}

/* **********************************************************
 @ computation
   ********************************************************** */
void compute_fftz_pack1(double *out, struct _offt_plan *po, int tile_ind, int myT)
{
  double *t = po->t;
  double mpi_t = 0.0;
  //t[PACK1] -= MPI_Wtime();
  struct _offt_buffer** buffers = (struct _offt_buffer**)(po->buffers1);
  struct _offt_comm *comm = po->comm;
  int T = po->params->v[_T1_]; int W = po->params->v[_W1_];
  int Nz = po->Nz;
  int Nz_new = (po->is_r2c)?(Nz/2+1):Nz;
#ifdef A2AV
  int F3 = comm->F3;
  int b3 = comm->b3;
  int m2 = comm->m2;
  int p2 = comm->p2;
  int is_a2av = po->params->v[_V_] & 2;
  int M2 = comm->M2; int M3 = comm->M3;
#else
  int M2 = comm->M2; int M3 = comm->M3;
#endif
  int from_x = tile_ind * T;
  int to_x = from_x + myT;
  struct _offt_buffer *buffer = buffers[tile_ind % (W + 1)];

  int Px1 = po->params->v[_Px1_];
  int Py1 = po->params->v[_Py1_];
  int Fz = po->params->v[_Fz_];
  int FP1 = po->params->v[_FP1_];
  //int num_pencil = T / Px1 * comm->m2 / Py1; // MPI_Test
  int num_pencil = T * comm->M2;
  /* in case F values are too big */
  Fz = min(num_pencil, Fz);
  FP1 = min(num_pencil, FP1);
  int test_min_tile_ind = max(tile_ind - W, 0);
  int test_max_tile_ind = max(tile_ind - 1, 0);
  int test_curr_tile_ind = test_min_tile_ind;
  int pencil_count = 0;
  int test_count = 0;
  int pack_pencil_count = 0;
  int pack_test_count = 0;
  // loop tiling along x and y dimensions
  int xx, yy;
  for (xx = from_x; xx < to_x; xx += Px1) {
    int x_end = min(to_x, xx + Px1);
    for(yy = 0; yy < comm->m2; yy += Py1) {
      int y_end = min(comm->m2, yy + Py1);
      // for each sub-tile
      // FFTz
      int x, y, z;
      for (x = xx; x < x_end; x++) {
        for (y = yy; y < y_end; y++) { // Mm
          mpi_t = MPI_Wtime();
          t[PACK1] += mpi_t;
          t[FFTz] -= mpi_t;
          double *ptr = out + 2*comm->istride[1]*y + 2*comm->istride[0]*x;
          if (po->is_r2c) {
            fftw_execute_dft_r2c(po->p1d_z, (double*)ptr, (fftw_complex*)ptr);
          } else {
            fftw_execute_dft(po->p1d_z, (fftw_complex*)ptr, (fftw_complex*)ptr);
          }
          pencil_count++;
          mpi_t = MPI_Wtime();
          t[FFTz] += mpi_t;
          t[TEST1] -= mpi_t;
          if (
            W > 0 &&
            Fz > 0 &&
            tile_ind > 0 &&
            test_count < Fz &&
            num_pencil * (test_count + 1) / Fz == pencil_count)
          {
//if (!po->rank) { printf("test_count %d Fz %d pencil_count %d num_pencil %d\n", test_count, Fz, pencil_count, num_pencil); } // TODEL
            communicate_test(buffers[test_curr_tile_ind % (W + 1)]);
            test_curr_tile_ind++;
            if (test_curr_tile_ind > test_max_tile_ind)
              test_curr_tile_ind = test_min_tile_ind;
            test_count++;
          }
          mpi_t = MPI_Wtime();
          t[PACK1] -= mpi_t;
          t[TEST1] += mpi_t;
        }
      } // end of x
#ifdef STRIDE
      if (po->params->v[_S_]) {
#ifdef A2AV
        for (x = xx; x < x_end; x++) {
          for (y = yy; y < y_end; y++) {
            for (z = 0; z < Nz_new; z++) {
              int a, B, z_off, Sy, Sx;
              /* a: target rank index comm->ranks1[] determined by z */
              /* B: begin address in buffer for a reflecting T */
              /* z_off: begin address on z-dim in out for a */
              /* Sz: z-stride in buffer: F2 */
              /* Sx: x-stride in buffer: F2 * F3 */
              if (is_a2av) {
                if (F3*(p2-b3) <= z) {
                  a = (z - F3*(p2-b3)) / (F3+1) + (p2-b3);
                  B = (p2-b3)*(myT*m2*F3) + (a-(p2-b3))*(myT*m2*(F3+1));
                  z_off = (p2-b3)*F3 + (a-(p2-b3))*(F3+1);
                  Sy = F3+1;
                  Sx = m2 * (F3+1);
                } else {
                  a = z / F3;
                  B = a*(myT*m2*F3);
                  z_off = a*F3;
                  Sy = F3;
                  Sx = m2 * F3;
                }
              } else {
                if (F3*(p2-b3) <= z) {
                  a = (z - F3*(p2-b3)) / (F3+1) + (p2-b3);
                  B = a*(myT*M2*M3);
                  z_off = (p2-b3)*F3 + (a-(p2-b3))*(F3+1);
                  Sy = M3;
                  Sx = M2 * M3;
                } else {
                  a = z / F3;
                  B = a*(myT*M2*M3);
                  z_off = a*F3;
                  Sy = M3;
                  Sx = M2 * M3;
                }
              }
              memcpy(
                buffer->a2as + 2*(B + (z-z_off) + y*Sy + (x-from_x)*Sx),
                out + 2*z + 2*comm->istride[1]*y + 2*comm->istride[0]*x,
                2 * sizeof(double));
            }
          }
        }
#else // A2AV
        // pack: xyz => xyzB
        for (x = xx; x < x_end; x++) {
          for (y = yy; y < y_end; y++) { // Mm
            for (z = 0; z < Nz_new; z++) {
              memcpy(
                buffer->a2as + 2*((z/M3)*(M3*M2*myT) + (z%M3) + M3 * y + M2*M3*(x - from_x)),
                out + 2*z + 2*comm->istride[1]*y + 2*comm->istride[0]*x,
                2 * sizeof(double));
            }
  #if 0
            // chunk copy of z by M3
            for (z = 0; z < Nz_new; z+=M3) {
              memcpy(
                buffer->a2as + 2*((z/M3)*(M3*M2*myT) + (z%M3) + M3 * y + M2*M3*(x - from_x)),
                out + 2*z + 2*(M3*p2)*y + 2*yz_plane_size*x,
                2 * min(M3, Nz_new-z) * sizeof(double));
            }
  #endif
          }
        } // end of x
#endif // A2AV
      } else {
        // pack: xyz => xzyB
        /* loop order: destination xzyB */
        for (x = xx; x < x_end; x++) {
          for (z = 0; z < Nz_new; z++) {
#ifdef A2AV
            int a, B, z_off, Sz, Sx;
            /* a: target rank index comm->ranks1[] determined by z */
            /* B: begin address in buffer for a reflecting T */
            /* z_off: begin address on z-dim in out for a */
            /* Sz: z-stride in buffer: F2 */
            /* Sx: x-stride in buffer: F2 * F3 */
            if (is_a2av) {
              if (F3*(p2-b3) <= z) {
                a = (z - F3*(p2-b3)) / (F3+1) + (p2-b3);
                B = (p2-b3)*(myT*m2*F3) + (a-(p2-b3))*(myT*m2*(F3+1));
                z_off = (p2-b3)*F3 + (a-(p2-b3))*(F3+1);
                Sz = m2;
                Sx = m2 * (F3+1);
              } else {
                a = z / F3;
                B = a*(myT*m2*F3);
                z_off = a*F3;
                Sz = m2;
                Sx = m2 * F3;
              }
            } else {
              if (F3*(p2-b3) <= z) {
                a = (z - F3*(p2-b3)) / (F3+1) + (p2-b3);
                B = a*(myT*M2*M3);
                z_off = (p2-b3)*F3 + (a-(p2-b3))*(F3+1);
                Sz = M2;
                Sx = M2 * M3;
              } else {
                a = z / F3;
                B = a*(myT*M2*M3);
                z_off = a*F3;
                Sz = M2;
                Sx = M2 * M3;
              }
            }
#endif // A2AV
            for (y = yy; y < y_end; y++) { // Mm
#ifdef A2AV
  /*if (po->rank == 3) {
    printf("B %d z_off %d from_x %d Sz %d Sx %d y %d z %d x %d\n", B, z_off, from_x, Sz, Sx, y, z, x);
    fflush(stdout);
  }*/
              memcpy(
                buffer->a2as + 2*(B + y + (z-z_off)*Sz + (x-from_x)*Sx),
                out + 2*z + 2*comm->istride[1]*y + 2*comm->istride[0]*x,
                2 * sizeof(double));
#else // A2AV
              memcpy(
                buffer->a2as + 2*((z/M3)*(M3*M2*myT) + y + (z%M3)*M2 + M2*M3*(x - from_x)),
                out + 2*z + 2*comm->istride[1]*y + 2*comm->istride[0]*x,
                2 * sizeof(double));
#endif // A2AV
            }
          }
        } // end of x
      } /* end of if stride */
#else // STRIDE
      // pack: xyz => xzyB
      /* loop order: destination xzyB */
      for (x = xx; x < x_end; x++) {
        for (z = 0; z < Nz_new; z++) {
#ifdef A2AV
          int a, B, z_off, Sz, Sx;
          /* a: target rank index comm->ranks1[] determined by z */
          /* B: begin address in buffer for a reflecting T */
          /* z_off: begin address on z-dim in out for a */
          /* Sz: z-stride in buffer: F2 */
          /* Sx: x-stride in buffer: F2 * F3 */
          if (is_a2av) {
            if (F3*(p2-b3) <= z) {
              a = (z - F3*(p2-b3)) / (F3+1) + (p2-b3);
              B = (p2-b3)*(myT*m2*F3) + (a-(p2-b3))*(myT*m2*(F3+1));
              z_off = (p2-b3)*F3 + (a-(p2-b3))*(F3+1);
              Sz = m2;
              Sx = m2 * (F3+1);
            } else {
              a = z / F3;
              B = a*(myT*m2*F3);
              z_off = a*F3;
              Sz = m2;
              Sx = m2 * F3;
            }
          } else {
            if (F3*(p2-b3) <= z) {
              a = (z - F3*(p2-b3)) / (F3+1) + (p2-b3);
              B = a*(myT*M2*M3);
              z_off = (p2-b3)*F3 + (a-(p2-b3))*(F3+1);
              Sz = M2;
              Sx = M2 * M3;
            } else {
              a = z / F3;
              B = a*(myT*M2*M3);
              z_off = a*F3;
              Sz = M2;
              Sx = M2 * M3;
            }
          }
#endif // A2AV
          for (y = yy; y < y_end; y++) { // Mm
#ifdef A2AV
/*if (po->rank == 3) {
  printf("B %d z_off %d from_x %d Sz %d Sx %d y %d z %d x %d\n", B, z_off, from_x, Sz, Sx, y, z, x);
  fflush(stdout);
}*/
            memcpy(
              buffer->a2as + 2*(B + y + (z-z_off)*Sz + (x-from_x)*Sx),
              out + 2*z + 2*comm->istride[1]*y + 2*comm->istride[0]*x,
              2 * sizeof(double));
#else // A2AV
            memcpy(
              buffer->a2as + 2*((z/M3)*(M3*M2*myT) + y + (z%M3)*M2 + M2*M3*(x - from_x)),
              out + 2*z + 2*comm->istride[1]*y + 2*comm->istride[0]*x,
              2 * sizeof(double));
#endif // A2AV
          }
        }
      } // end of x
#endif // STRIDE
      pack_pencil_count = pencil_count;
      mpi_t = MPI_Wtime();
      t[TEST1] -= mpi_t;
      t[PACK1] += mpi_t;
      if (
        W > 0 &&
        FP1 > 0 &&
        tile_ind > 0 &&
        pack_test_count < FP1 &&
        num_pencil * (pack_test_count + 1) / FP1 == pack_pencil_count)
      {
//if (!po->rank) { printf("pack_test_count %d FP1 %d pack_pencil_count %d num_pencil %d\n", pack_test_count, FP1, pack_pencil_count, num_pencil); } // TODEL
        communicate_test(buffers[test_curr_tile_ind % (W + 1)]);
        test_curr_tile_ind++;
        if (test_curr_tile_ind > test_max_tile_ind)
          test_curr_tile_ind = test_min_tile_ind;
        pack_test_count++;
      }
      mpi_t = MPI_Wtime();
      t[PACK1] -= mpi_t;
      t[TEST1] += mpi_t;
    } // end of yy
  } // end of xx
  //t[PACK1] += MPI_Wtime();
}

void compute_unpack1_ffty(double *out, struct _offt_plan *po, int tile_ind, int myT)
{
  double *t = po->t;
  double mpi_t = 0.0;
  //t[UNPACK1] -= MPI_Wtime();
  struct _offt_buffer** buffers = (struct _offt_buffer**)(po->buffers1);
  struct _offt_comm* comm = po->comm;
  int T = po->params->v[_T1_];
  int W = po->params->v[_W1_];
#ifdef A2AV
  int F2 = comm->F2;
  int b2 = comm->b2;
  int M3 = comm->M3; int M4 = comm->M4;
  int m3 = comm->m3;
  int p1 = comm->p1; int p2 = comm->p2;
  int is_a2av = po->params->v[_V_] & 2;
  int M2 = comm->M2;
#else
  int M2 = comm->M2;
  int M3 = comm->M3;
  int p1 = comm->p1;
  int M4 = comm->M4;
#endif
  int Ny = po->Ny;
  int from_x = tile_ind * T;
  int to_x = from_x + myT;

  struct _offt_buffer* buffer = buffers[tile_ind % (W + 1)];

  int Ry = po->params->v[_Ry_];
  //int Ry_count = 0;
  /* ignore Ry for is_oned, 1 x p case */
  int is_ignore_Ry = (po->is_oned && po->comm->p1 == 1);
  int Ux1 = po->params->v[_Ux1_];
  int Uz1 = po->params->v[_Uz1_];
  int FU1 = po->params->v[_FU1_];
  int Fy1 = po->params->v[_Fy1_];
  //int num_pencil = T / Ux1 * comm->m3 / Uz1;
  int num_pencil = T * comm->M3;
  /* in case F values are too big */
  FU1 = min(num_pencil, FU1);
  Fy1 = min(num_pencil, Fy1);
  int unpack_pencil_count = 0;
  int pencil_count = 0;
  int unpack_test_count = 0;
  int test_count = 0;
  int last_tile_ind = (comm->m1 - 1) / T;
  int test_min_tile_ind = min(tile_ind + 1, last_tile_ind);
  int test_max_tile_ind = min(tile_ind + W, last_tile_ind);
  int test_curr_tile_ind = test_min_tile_ind;
  // loop tiling along x and z dimensions
  int xx, zz;
  for(xx = from_x; xx < to_x; xx += Ux1) {
    int x_end = min(to_x, xx + Ux1);
    for (zz = 0; zz < comm->m3; zz += Uz1) {
      int z_end = min(comm->m3, zz + Uz1);
      int x, z, y;
#ifdef STRIDE
      if (po->params->v[_S_]) {
#ifdef A2AV
        // unpack: xyzB => xyz
        for (x = xx; x < x_end; x++) {
          for (y = 0; y < Ny; y++) {
            for (z = zz; z < z_end; z++) { // Mm
              int a, B, y_off, Sy, Sx;
              /* a: source rank index comm->ranks1[] determined by y */
              /* B: begin address in buffer for a reflecting T */
              /* y_off: begin address on y-dim in buffer for a */
              /* Sz: z-stride in buffer: F2 */
              /* Sx: x-stride in buffer: F2 * m3_ */
              if (is_a2av) {
                if (F2*(p2-b2) <= y) {
                  a = (y - F2*(p2-b2)) / (F2+1) + (p2-b2);
                  B = (p2-b2)*(myT*F2*m3) + (a-(p2-b2))*(myT*(F2+1)*m3);
                  y_off = (p2-b2)*F2 + (a-(p2-b2))*(F2+1);
                  Sy = m3;
                  Sx = (F2+1) * m3;
                } else {
                  a = y / F2;
                  B = a*(myT*F2*m3);
                  y_off = a*F2;
                  Sy = m3;
                  Sx = F2 * m3;
                }
              } else {
                if (F2*(p2-b2) <= y) {
                  a = (y - F2*(p2-b2)) / (F2+1) + (p2-b2);
                  B = a*(myT*M2*M3);
                  y_off = (p2-b2)*F2 + (a-(p2-b2))*(F2+1);
                  Sy = M3;
                  Sx = M2 * M3;
                } else {
                  a = y / F2;
                  B = a*(myT*M2*M3);
                  y_off = a*F2;
                  Sy = M3;
                  Sx = M2 * M3;
                }
              }
              memcpy(
                // TODO mstride
                out + 2*(z + M3*y + M3*M4*p1 * x), // M4*p1 >= Ny
                buffer->a2ar + 2*(B + z + (y-y_off)*Sy + (x - from_x)*Sx),
                2 * sizeof(double)); /* for Ny%p2=0 and Ny%p1!=0 */
            }
          }
          unpack_pencil_count += (z_end - zz);
        }
#else // A2AV
        // unpack: xyzB => xyz
        for (x = xx; x < x_end; x++) {
          for (y = 0; y < Ny; y++) {
            for (z = zz; z < z_end; z++) { // Mm
              memcpy(
                out + 2*(z + M3*(y + M4*p1*x)),
                buffer->a2ar + 2*((y/M2)*(M3*M2*myT) + z + M3 * (y%M2) + M2*M3*(x - from_x)),
                2 * sizeof(double));
            }
  #if 0
            // chunk copy z by M3
            z = zz;
            memcpy(
              out + 2*(z + M3*(y + M4*p1*x)),
              buffer->a2ar + 2*((y/M2)*(M3*M2*myT) + z + M3 * (y%M2) + M2*M3*(x - from_x)),
              2 * (z_end - zz) * sizeof(double));
  #endif
          }
          unpack_pencil_count += (z_end - zz);
        }
#endif
      } else {
        // unpack: xzyB => xzy
        /* loop order: destination xzy */
        for (x = xx; x < x_end; x++) {
          for (z = zz; z < z_end; z++) { // Mm
            /* consider yB (M2 complex numbers on y dim) as one element. */
#ifdef A2AV
            /* TODO: chunk copy of every m2 elements on y-dim */
            for (y = 0; y < Ny; y++) {
              int a, B, y_off, Sz, Sx;
              /* a: source rank index comm->ranks1[] determined by y */
              /* B: begin address in buffer for a reflecting T */
              /* y_off: begin address on y-dim in buffer for a */
              /* Sz: z-stride in buffer: F2 */
              /* Sx: x-stride in buffer: F2 * m3_ */
              if (is_a2av) {
                if (F2*(p2-b2) <= y) {
                  a = (y - F2*(p2-b2)) / (F2+1) + (p2-b2);
                  B = (p2-b2)*(myT*F2*m3) + (a-(p2-b2))*(myT*(F2+1)*m3);
                  y_off = (p2-b2)*F2 + (a-(p2-b2))*(F2+1);
                  Sz = F2+1;
                  Sx = (F2+1) * m3;
                } else {
                  a = y / F2;
                  B = a*(myT*F2*m3);
                  y_off = a*F2;
                  Sz = F2;
                  Sx = F2 * m3;
                }
              } else {
                if (F2*(p2-b2) <= y) {
                  a = (y - F2*(p2-b2)) / (F2+1) + (p2-b2);
                  B = a*(myT*M2*M3);
                  y_off = (p2-b2)*F2 + (a-(p2-b2))*(F2+1);
                  Sz = M2;
                  Sx = M2 * M3;
                } else {
                  a = y / F2;
                  B = a*(myT*M2*M3);
                  y_off = a*F2;
                  Sz = M2;
                  Sx = M2 * M3;
                }
              }
              memcpy(
                out + 2*(y + M4*p1*(z + M3 * x)), // M4*p1 >= Ny
                buffer->a2ar + 2*(B + (y-y_off) + z*Sz + (x - from_x)*Sx),
                2 * sizeof(double)); /* for Ny%p2=0 and Ny%p1!=0 */
            }
#else // A2AV
            for (y = 0; y < Ny; y+=M2) {
              memcpy(
                out + 2*(y + M4*p1*(z + M3 * x)), // M4*p1 >= Ny
                buffer->a2ar + 2*((y/M2)*(M3*M2*myT) + (y%M2) + z*M2 + M2*M3*(x - from_x)),
                2 * min(M2, Ny-y) * sizeof(double)); /* for Ny%p2=0 and Ny%p1!=0 */
            }
#endif // A2AV
            unpack_pencil_count++;
          }
        }
      } /* end of if stride */
#else // STRIDE
      // unpack: xzyB => xzy
      /* loop order: destination xzy */
      for (x = xx; x < x_end; x++) {
        for (z = zz; z < z_end; z++) { // Mm
          /* consider yB (M2 complex numbers on y dim) as one element. */
#ifdef A2AV
          /* TODO: chunk copy of every m2 elements on y-dim */
          for (y = 0; y < Ny; y++) {
            int a, B, y_off, Sz, Sx;
            /* a: source rank index comm->ranks1[] determined by y */
            /* B: begin address in buffer for a reflecting T */
            /* y_off: begin address on y-dim in buffer for a */
            /* Sz: z-stride in buffer: F2 */
            /* Sx: x-stride in buffer: F2 * m3_ */
            if (is_a2av) {
              if (F2*(p2-b2) <= y) {
                a = (y - F2*(p2-b2)) / (F2+1) + (p2-b2);
                B = (p2-b2)*(myT*F2*m3) + (a-(p2-b2))*(myT*(F2+1)*m3);
                y_off = (p2-b2)*F2 + (a-(p2-b2))*(F2+1);
                Sz = F2+1;
                Sx = (F2+1) * m3;
              } else {
                a = y / F2;
                B = a*(myT*F2*m3);
                y_off = a*F2;
                Sz = F2;
                Sx = F2 * m3;
              }
            } else {
              if (F2*(p2-b2) <= y) {
                a = (y - F2*(p2-b2)) / (F2+1) + (p2-b2);
                B = a*(myT*M2*M3);
                y_off = (p2-b2)*F2 + (a-(p2-b2))*(F2+1);
                Sz = M2;
                Sx = M2 * M3;
              } else {
                a = y / F2;
                B = a*(myT*M2*M3);
                y_off = a*F2;
                Sz = M2;
                Sx = M2 * M3;
              }
            }
            memcpy(
              out + 2*(y + M4*p1*(z + M3 * x)), // M4*p1 >= Ny
              buffer->a2ar + 2*(B + (y-y_off) + z*Sz + (x - from_x)*Sx),
              2 * sizeof(double)); /* for Ny%p2=0 and Ny%p1!=0 */
          }
#else // A2AV
          for (y = 0; y < Ny; y+=M2) {
            memcpy(
              out + 2*(y + M4*p1*(z + M3 * x)), // M4*p1 >= Ny
              buffer->a2ar + 2*((y/M2)*(M3*M2*myT) + (y%M2) + z*M2 + M2*M3*(x - from_x)),
              2 * min(M2, Ny-y) * sizeof(double)); /* for Ny%p2=0 and Ny%p1!=0 */
          }
#endif // A2AV
          unpack_pencil_count++;
        }
      }
#endif // STRIDE
      mpi_t = MPI_Wtime();
      t[TEST1] -= mpi_t;
      t[UNPACK1] += mpi_t;
      if (
        W > 0 &&
        FU1 > 0 &&
        unpack_test_count < FU1 &&
        num_pencil * (unpack_test_count + 1) / FU1 == unpack_pencil_count)
      {
        communicate_test(buffers[test_curr_tile_ind % (W + 1)]);
        test_curr_tile_ind++;
        if (test_curr_tile_ind > test_max_tile_ind)
          test_curr_tile_ind = test_min_tile_ind;
        unpack_test_count++;
      }
      mpi_t = MPI_Wtime();
      t[UNPACK1] -= mpi_t;
      t[TEST1] += mpi_t;
      // FFTy
      for (x = xx; x < x_end; x++) {
        for (z = zz; z < z_end; z++) { // Mm
          mpi_t = MPI_Wtime();
          t[UNPACK1] += mpi_t;
          t[FFTy1] -= mpi_t;
          if (is_ignore_Ry || x % 10 < Ry) {
#ifdef STRIDE
            double *ptr = NULL;
            if (po->params->v[_S_])
              ptr = out + 2*(z + M4*p1*M3 * x);
            else
              ptr = out + 2*M4*p1*(z + M3 * x);
#else // STRIDE
            double *ptr = out + 2*M4*p1*(z + M3 * x);
#endif // STRIDE
            fftw_execute_dft(po->p1d_y, (fftw_complex*)ptr, (fftw_complex*)ptr);
          }
          pencil_count++;
          mpi_t = MPI_Wtime();
          t[FFTy1] += mpi_t;
          t[TEST1] -= mpi_t;
          if (
            W > 0 &&
            Fy1 > 0 &&
            test_count < Fy1 &&
            num_pencil * (test_count + 1) / Fy1 == pencil_count)
          {
            communicate_test(buffers[test_curr_tile_ind % (W + 1)]);
            test_curr_tile_ind++;
            if (test_curr_tile_ind > test_max_tile_ind)
              test_curr_tile_ind = test_min_tile_ind;
            test_count++;
          }
          mpi_t = MPI_Wtime();
          t[UNPACK1] -= mpi_t;
          t[TEST1] += mpi_t;
        } // end of z
      } // end of x
    } // end of zz
  } // end of xx
  //t[UNPACK1] += MPI_Wtime();
}

#if 0
void compute_pack2(double *out, struct _offt_plan* po, int tile_ind, int myT)
{
  double *t = po->t;
  struct _offt_buffer** buffers = (struct _offt_buffer**)(po->buffers);
  struct _offt_comm* comm = po->comm;
  int T = po->params->v[_T2_]; int W = po->params->v[_W2_];
  int M1 = comm->M1;
  int M3 = comm->M3;
  int M4 = comm->M4;
  int p1 = comm->p1;
  int Ny = po->Ny;
  int from_z = tile_ind * T;
  int to_z = min(comm->m3, from_z + T); // Mm
  struct _offt_buffer* buffer = buffers[tile_ind % (W + 1)];

  int Pz2 = po->params->v[_Pz2_];
  int Px2 = po->params->v[_Px2_];
  int Py2 = po->params->v[_Py2_];
  int FP2 = po->params->v[_FP2_];
  int num_subtile = T / Pz2 * comm->m1 / Px2 * Ny / Py2;
  int test_min_tile_ind = max(tile_ind - W, 0);
  int test_max_tile_ind = max(tile_ind - 1, 0);
  int test_curr_tile_ind = test_min_tile_ind;
  int subtile_count = 0;
  int test_count = 0;
  /* loop tiling along x, y, and z dimensions */
  int zz, xx, yy;
  if (po->is_equalxy && M1 == M4) {
    /* xzy->yzxB */
    /* outer loop order: source xzy vs yzx? */
    for(xx = 0; xx < comm->m1; xx += Px2) {
      int x_end = min(comm->m1, xx + Px2);
      for (zz = from_z; zz < to_z; zz += Pz2) {
        int z_end = min(to_z, zz + Pz2);
        for(yy = 0; yy < Ny; yy += Py2) {
          int y_end = min(Ny, yy + Py2);
          /* pack: xzy => yzxB */
          t[PACK2] -= MPI_Wtime();
          int z, x, y;
          /* loop order: destination yzxB */
          for (y = yy; y < y_end; y++) {
            for (z = zz; z < z_end; z++) {
              for (x = xx; x < x_end; x++) {
                memcpy(
                  buffer->a2as + 2*((y/M4)*(M4*M1*T) + x + (z-from_z)*M1 + (y%M4)*M1*T),
                  out + 2*(y + (M4*p1)*(z + M3 * x)),
                  2 * sizeof(double));
              }
            }
          }
          subtile_count++;
          t[PACK2] += MPI_Wtime();
          if (
            W > 0 &&
            FP2 > 0 &&
            tile_ind > 0 &&
            test_count < FP2 &&
            num_subtile * (test_count + 1) / FP2 == subtile_count)
          {
            communicate_test(po, buffers[test_curr_tile_ind % (W + 1)], 0);
            test_curr_tile_ind++;
            if (test_curr_tile_ind > test_max_tile_ind)
              test_curr_tile_ind = test_min_tile_ind;
            test_count++;
          }
        } // end of yy
      } // end of xx
    } // end of zz
  } else {
    /* zxy->zyxB */
    /* outer loop order: source zxy 0.24s vs destination zyx 0.27s why?? */
    for (zz = from_z; zz < to_z; zz += Pz2) {
      int z_end = min(to_z, zz + Pz2);
      for(xx = 0; xx < comm->m1; xx += Px2) {
        int x_end = min(comm->m1, xx + Px2);
        for(yy = 0; yy < Ny; yy += Py2) {
          int y_end = min(Ny, yy + Py2);
          /* pack: zxy => zyxB */
          t[PACK2] -= MPI_Wtime();
          int z, x, y;
          /* loop order: destination zyxB */
          for (z = zz; z < z_end; z++) {
            for (y = yy; y < y_end; y++) {
              for (x = xx; x < x_end; x++) { // Mm
                memcpy(
                  buffer->a2as + 2*((y/M4)*(M4*M1*T) + x + (y%M4)*M1 + (z-from_z)*M1*M4),
                  out + 2*(y + (M4*p1)*(x + M1 * z)),
                  2 * sizeof(double));
              }
            }
          }
          subtile_count++;
          t[PACK2] += MPI_Wtime();
          if (
            W > 0 &&
            FP2 > 0 &&
            tile_ind > 0 &&
            test_count < FP2 &&
            num_subtile * (test_count + 1) / FP2 == subtile_count)
          {
            communicate_test(po, buffers[test_curr_tile_ind % (W + 1)], 0);
            test_curr_tile_ind++;
            if (test_curr_tile_ind > test_max_tile_ind)
              test_curr_tile_ind = test_min_tile_ind;
            test_count++;
          }
        } // end of yy
      } // end of xx
    } // end of zz
  } /* end of if is_equalxy */
}
#endif

void compute_ffty_pack2(double *out, struct _offt_plan *po, int tile_ind, int myT)
{
  double *t = po->t;
  double mpi_t = 0.0;
  //t[PACK2] -= MPI_Wtime();
  struct _offt_buffer** buffers = (struct _offt_buffer**)(po->buffers2);
  struct _offt_comm *comm = po->comm;
  int T = po->params->v[_T2_]; int W = po->params->v[_W2_];
  int M3 = comm->M3;
  int M4 = comm->M4;
  int p1 = comm->p1;
#ifdef A2AV
  int F4 = comm->F4;
  int m1 = comm->m1;
  int b4 = comm->b4;
  int is_a2av = po->params->v[_V_] & 1;
#ifdef STRIDE
  int M1 = comm->M1;
#else
  int M1 = comm->M1;
#endif
#else
  int M1 = comm->M1;
#endif
  int Ny = po->Ny;
  int from_z = tile_ind * T;
  int to_z = from_z + myT;
  struct _offt_buffer* buffer = buffers[tile_ind % (W + 1)];

  int Px2 = po->params->v[_Px2_];
  int Pz2 = po->params->v[_Pz2_];
  /* ignore Py2 */
  int Ry = po->params->v[_Ry_];
  //int Ry_count = 0;
  int Fy2 = po->params->v[_Fy2_];
  int FP2 = po->params->v[_FP2_];
  int num_pencil = T * comm->M1;
  /* in case F values are too big */
  Fy2 = min(num_pencil, Fy2);
  FP2 = min(num_pencil, FP2);
  int test_min_tile_ind = max(tile_ind - W, 0);
  int test_max_tile_ind = max(tile_ind - 1, 0);
  int test_curr_tile_ind = test_min_tile_ind;
  int pencil_count = 0;
  int test_count = 0;
  int pack_pencil_count = 0;
  int pack_test_count = 0;
#ifdef FINETEST
  int num_xyplane = po->Ny * comm->m1 / Px2 * myT / Pz2;
  int pack_xyplane_count = 0;
  FP2 = min(num_xyplane, po->params->v[_FP2_]);
#endif
  
  /* ignore Ry for is_oned, p x 1 case */
  int is_ignore_Ry = (po->is_oned && po->comm->p1 == po->p);

  /* loop tiling along z and x dimensions */
  int zz, xx;
#ifdef STRIDE
  if (po->params->v[_S_]) {
    for(xx = 0; xx < comm->m1; xx += Px2) {
      int x_end = min(comm->m1, xx + Px2);
      for (zz = from_z; zz < to_z; zz += Pz2) {
        int z_end = min(to_z, zz + Pz2);
        // for each sub-tile
        // FFTy
        int x, y, z;
        for (x = xx; x < x_end; x++) {
          for (z = zz; z < z_end; z++) {
            mpi_t = MPI_Wtime();
            t[PACK2] += mpi_t;
            t[FFTy2] -= mpi_t;
            if (is_ignore_Ry || x % 10 >= Ry) {
              double *ptr = out + 2*z + 2*M4*p1*(M3 * x);
              fftw_execute_dft(po->p1d_y, (fftw_complex*)ptr, (fftw_complex*)ptr);
            }
            pencil_count++;
            mpi_t = MPI_Wtime();
            t[FFTy2] += mpi_t;
            t[TEST2] -= mpi_t;
            if (
              W > 0 &&
              Fy2 > 0 &&
              tile_ind > 0 &&
              test_count < Fy2 &&
              num_pencil * (test_count + 1) / Fy2 == pencil_count)
            {
              communicate_test(buffers[test_curr_tile_ind % (W + 1)]);
              test_curr_tile_ind++;
              if (test_curr_tile_ind > test_max_tile_ind)
                test_curr_tile_ind = test_min_tile_ind;
              test_count++;
            }
            mpi_t = MPI_Wtime();
            t[PACK2] -= mpi_t;
            t[TEST2] += mpi_t;
          }
        } // end of x
#ifdef A2AV
        for (x = xx; x < x_end; x++) {
          for (y = 0; y < Ny; y++) {
            for (z = zz; z < z_end; z++) {
              int a, B, y_off, Sy, Sx;
              /* a: target rank index comm->ranks2[] determined by y */
              /* B: begin address in buffer for a reflecting T */
              /* y_off: begin address on y-dim in out for a */
              /* Sy: y-stride in buffer: myT */
              /* Sx: x-stride in buffer: myT * (F4+1) */
              if (is_a2av) {
                if (F4*(p1-b4) <= y) {
                  a = (y - F4*(p1-b4)) / (F4+1) + (p1-b4);
                  B = (p1-b4)*(m1*F4*myT) + (a-(p1-b4))*(m1*(F4+1)*myT);
                  y_off = (p1-b4)*F4 + (a-(p1-b4))*(F4+1);
                  Sy = myT;
                  Sx = myT * (F4+1);
                } else {
                  a = y / F4;
                  B = a*(m1*F4*myT);
                  y_off = a*F4;
                  Sy = myT;
                  Sx = myT * F4;
                }
              } else {
                if (F4*(p1-b4) <= y) {
                  a = (y - F4*(p1-b4)) / (F4+1) + (p1-b4);
                  B = a*(M1*M4*myT);
                  y_off = (p1-b4)*F4 + (a-(p1-b4))*(F4+1);
                  Sy = myT;
                  Sx = myT * M4;
                } else {
                  a = y / F4;
                  B = a*(M1*M4*myT);
                  y_off = a*F4;
                  Sy = myT;
                  Sx = myT * M4;
                }
              }
              memcpy(
                buffer->a2as + 2*(B + (z-from_z) + (y-y_off)*Sy + x*Sx),
                out + 2*z + 2*M3*y + 2*M3*M4*p1*x,
                2 * sizeof(double));
            }
          }
        }
#else // A2AV
        // pack: xyz => xyzB
        for (x = xx; x < x_end; x++) {
          for (y = 0; y < Ny; y++) {
            //for (z = zz; z < z_end; z++) {
              memcpy(
                buffer->a2as + 2*((y/M4)*(M4*M1*myT) + (zz-from_z) + (y%M4)*myT + x*M4*myT),
                out + 2*(zz + M3*(y + M4*p1 * x)),
                (z_end-zz) * 2 * sizeof(double));
            //}
          }
        } /* end of y */
#if 0
        // pack: xyz => yxzB
        for (y = 0; y < Ny; y++) {
          for (x = xx; x < x_end; x++) {
            //for (z = zz; z < z_end; z++) {
              memcpy(
                buffer->a2as + 2*((y/M4)*(M4*M1*myT) + (zz-from_z) + x*myT + (y%M4)*M1*myT),
                out + 2*(zz + M3*(y + M4*p1 * x)),
                (z_end-zz) * 2 * sizeof(double));
            //}
          }
        } /* end of y */
#endif
#endif // A2AV
        pack_pencil_count = pencil_count;
        mpi_t = MPI_Wtime();
        t[TEST2] -= mpi_t;
        t[PACK2] += mpi_t;
        if (
          W > 0 &&
          FP2 > 0 &&
          tile_ind > 0 &&
          pack_test_count < FP2 &&
          num_pencil * (pack_test_count + 1) / FP2 == pack_pencil_count)
        {
          communicate_test(buffers[test_curr_tile_ind % (W + 1)]);
          test_curr_tile_ind++;
          if (test_curr_tile_ind > test_max_tile_ind)
            test_curr_tile_ind = test_min_tile_ind;
          pack_test_count++;
        }
        t[PACK2] -= mpi_t;
        t[TEST2] += mpi_t;
      } // end of xx
    } // end of zz
  } else { // if stride
    if (po->is_equalxy && comm->M1 == comm->M4) {
      for(xx = 0; xx < comm->m1; xx += Px2) {
        int x_end = min(comm->m1, xx + Px2);
        for (zz = from_z; zz < to_z; zz += Pz2) {
          int z_end = min(to_z, zz + Pz2);
          // for each sub-tile
          // FFTy
          int x, y, z;
          for (x = xx; x < x_end; x++) {
            for (z = zz; z < z_end; z++) {
              mpi_t = MPI_Wtime();
              t[PACK2] += mpi_t;
              t[FFTy2] -= mpi_t;
              if (is_ignore_Ry || x % 10 >= Ry) {
                double *ptr = out + 2*M4*p1*(z + M3 * x);
                fftw_execute_dft(po->p1d_y, (fftw_complex*)ptr, (fftw_complex*)ptr);
              }
              pencil_count++;
              mpi_t = MPI_Wtime();
              t[FFTy2] += mpi_t;
              t[TEST2] -= mpi_t;
              if (
                W > 0 &&
                Fy2 > 0 &&
                tile_ind > 0 &&
                test_count < Fy2 &&
                num_pencil * (test_count + 1) / Fy2 == pencil_count)
              {
                communicate_test(buffers[test_curr_tile_ind % (W + 1)]);
                test_curr_tile_ind++;
                if (test_curr_tile_ind > test_max_tile_ind)
                  test_curr_tile_ind = test_min_tile_ind;
                test_count++;
              }
              mpi_t = MPI_Wtime();
              t[PACK2] -= mpi_t;
              t[TEST2] += mpi_t;
            }
          } // end of x
  #ifdef A2AV
          // pack: xzy => yzxB
          /* loop order: destination yzxB */
          for (y = 0; y < Ny; y++) {
            for (z = zz; z < z_end; z++) {
              for (x = xx; x < x_end; x++) {
                int a, B, y_off, Sz, Sy;
                /* a: target rank index comm->ranks2[] determined by y */
                /* B: begin address in buffer for a reflecting T */
                /* y_off: begin address on y-dim in out for a */
                /* Sy: y-stride in buffer: myT */
                /* Sx: x-stride in buffer: myT * (F4+1) */
                if (is_a2av) {
                  if (F4*(p1-b4) <= y) {
                    a = (y - F4*(p1-b4)) / (F4+1) + (p1-b4);
                    B = (p1-b4)*(m1*F4*myT) + (a-(p1-b4))*(m1*(F4+1)*myT);
                    y_off = (p1-b4)*F4 + (a-(p1-b4))*(F4+1);
                    Sz = m1;
                    Sy = m1 * myT;
                  } else {
                    a = y / F4;
                    B = a*(m1*F4*myT);
                    y_off = a*F4;
                    Sz = m1;
                    Sy = m1 * myT;
                  }
                } else {
                  if (F4*(p1-b4) <= y) {
                    a = (y - F4*(p1-b4)) / (F4+1) + (p1-b4);
                    B = a*(M1*M4*myT);
                    y_off = (p1-b4)*F4 + (a-(p1-b4))*(F4+1);
                    Sz = M1;
                    Sy = M1 * myT;
                  } else {
                    a = y / F4;
                    B = a*(M1*M4*myT);
                    y_off = a*F4;
                    Sz = M1;
                    Sy = M1 * myT;
                  }
                }
                memcpy(
                  buffer->a2as + 2*(B + x + (z-from_z)*Sz + (y-y_off)*Sy),
                  out + 2*y + 2*Ny*z + 2*M4*p1*M3*x,
                  2 * sizeof(double));
              }
            }
#ifdef FINETEST
          pack_xyplane_count++;
          mpi_t = MPI_Wtime();
          t[TEST2] -= mpi_t;
          t[PACK2] += mpi_t;
          if (
            W > 0 &&
            FP2 > 0 &&
            tile_ind > 0 &&
            pack_test_count < FP2 &&
            num_xyplane * (pack_test_count + 1) / FP2 == pack_xyplane_count)
          {
//if (!po->rank) { printf("test %d : %d ; xyplane %d : %d\n", pack_test_count, FP2, pack_xyplane_count, num_xyplane); }
            communicate_test(buffers[test_curr_tile_ind % (W + 1)]);
            test_curr_tile_ind++;
            if (test_curr_tile_ind > test_max_tile_ind)
              test_curr_tile_ind = test_min_tile_ind;
            pack_test_count++;
          }
          t[PACK2] -= mpi_t;
          t[TEST2] += mpi_t;
#endif // FINETEST
          }
  #else // A2AV
          // pack: xzy => yzxB
          /* loop order: destination yzxB */
          for (y = 0; y < Ny; y++) {
            for (z = zz; z < z_end; z++) {
              for (x = xx; x < x_end; x++) {
                memcpy(
                  buffer->a2as + 2*((y/M4)*(M4*M1*myT) + x + (z-from_z)*M1 + (y%M4)*M1*myT),
                  out + 2*(y + (M4*p1)*(z + M3 * x)),
                  2 * sizeof(double));
              }
            }
          } /* end of y */
  #endif // A2AV
#ifdef FINETEST
#else // FINETEST
          pack_pencil_count = pencil_count;
          mpi_t = MPI_Wtime();
          t[TEST2] -= mpi_t;
          t[PACK2] += mpi_t;
          if (
            W > 0 &&
            FP2 > 0 &&
            tile_ind > 0 &&
            pack_test_count < FP2 &&
            num_pencil * (pack_test_count + 1) / FP2 == pack_pencil_count)
          {
            communicate_test(buffers[test_curr_tile_ind % (W + 1)]);
            test_curr_tile_ind++;
            if (test_curr_tile_ind > test_max_tile_ind)
              test_curr_tile_ind = test_min_tile_ind;
            pack_test_count++;
          }
          t[PACK2] -= mpi_t;
          t[TEST2] += mpi_t;
#endif // FINETEST
        } // end of xx
      } // end of zz
    } else {
      for (zz = from_z; zz < to_z; zz += Pz2) {
        int z_end = min(to_z, zz + Pz2);
        for(xx = 0; xx < comm->m1; xx += Px2) {
          int x_end = min(comm->m1, xx + Px2);
          // for each sub-tile
          // FFTy
          int x, y, z;
          for (z = zz; z < z_end; z++) {
            for (x = xx; x < x_end; x++) {
              mpi_t = MPI_Wtime();
              t[PACK2] += mpi_t;
              t[FFTy2] -= mpi_t;
              if (is_ignore_Ry || x % 10 >= Ry) {
                double *ptr = out + 2*M4*p1*(x + M1 * z);
                fftw_execute_dft(po->p1d_y, (fftw_complex*)ptr, (fftw_complex*)ptr);
              }
              pencil_count++;
              mpi_t = MPI_Wtime();
              t[FFTy2] += mpi_t;
              t[TEST2] -= mpi_t;
              if (
                W > 0 &&
                Fy2 > 0 &&
                tile_ind > 0 &&
                test_count < Fy2 &&
                num_pencil * (test_count + 1) / Fy2 == pencil_count)
              {
                communicate_test(buffers[test_curr_tile_ind % (W + 1)]);
                test_curr_tile_ind++;
                if (test_curr_tile_ind > test_max_tile_ind)
                  test_curr_tile_ind = test_min_tile_ind;
                test_count++;
              }
              mpi_t = MPI_Wtime();
              t[PACK2] -= mpi_t;
              t[TEST2] += mpi_t;
            }
          } // end of x
          // pack: zxy => zyxB
          /* loop order: destination zyxB */
  #ifdef A2AV
          for (z = zz; z < z_end; z++) {
            for (y = 0; y < Ny; y++) {
              for (x = xx; x < x_end; x++) {
                int a, B, y_off, Sz, Sy;
                /* a: target rank index comm->ranks2[] determined by y */
                /* B: begin address in buffer for a reflecting T */
                /* y_off: begin address on y-dim in out for a */
                /* Sy: y-stride in buffer: myT */
                /* Sx: x-stride in buffer: myT * (F4+1) */
                if (is_a2av) {
                  if (F4*(p1-b4) <= y) {
                    a = (y - F4*(p1-b4)) / (F4+1) + (p1-b4);
                    B = (p1-b4)*(m1*F4*myT) + (a-(p1-b4))*(m1*(F4+1)*myT);
                    y_off = (p1-b4)*F4 + (a-(p1-b4))*(F4+1);
                    Sy = m1;
                    Sz = m1 * (F4+1);
                  } else {
                    a = y / F4;
                    B = a*(m1*F4*myT);
                    y_off = a*F4;
                    Sy = m1;
                    Sz = m1 * F4;
                  }
                } else {
                  if (F4*(p1-b4) <= y) {
                    a = (y - F4*(p1-b4)) / (F4+1) + (p1-b4);
                    B = a*(M1*M4*myT);
                    y_off = (p1-b4)*F4 + (a-(p1-b4))*(F4+1);
                    Sy = M1;
                    Sz = M1 * M4;
                  } else {
                    a = y / F4;
                    B = a*(M1*M4*myT);
                    y_off = a*F4;
                    Sy = M1;
                    Sz = M1 * M4;
                  }
                }
                memcpy(
                  buffer->a2as + 2*(B + x + (y-y_off)*Sy + (z-from_z)*Sz),
                  out + 2*y + 2*M4*p1*x + 2*M4*p1*M1*z,
                  2 * sizeof(double));
              }
            }
          }
  #else // A2AV
          for (z = zz; z < z_end; z++) {
            for (y = 0; y < Ny; y++) {
              for (x = xx; x < x_end; x++) { // Mm
                memcpy(
                  buffer->a2as + 2*((y/M4)*(M4*M1*myT) + x + (y%M4)*M1 + (z-from_z)*M1*M4),
                  out + 2*(y + (M4*p1)*(x + M1 * z)),
                  2 * sizeof(double));
              }
            }
          } /* end of z */
  #endif // A2AV
          pack_pencil_count = pencil_count;
          mpi_t = MPI_Wtime();
          t[TEST2] -= mpi_t;
          t[PACK2] += mpi_t;
          if (
            W > 0 &&
            FP2 > 0 &&
            tile_ind > 0 &&
            pack_test_count < FP2 &&
            num_pencil * (pack_test_count + 1) / FP2 == pack_pencil_count)
          {
            communicate_test(buffers[test_curr_tile_ind % (W + 1)]);
            test_curr_tile_ind++;
            if (test_curr_tile_ind > test_max_tile_ind)
              test_curr_tile_ind = test_min_tile_ind;
            pack_test_count++;
          }
          mpi_t = MPI_Wtime();
          t[PACK2] -= mpi_t;
          t[TEST2] += mpi_t;
        } // end of xx
      } // end of zz
    } /* end of if is_equalxy */
  } /* end of if stride */
#else // STRIDE
  if (po->is_equalxy && comm->M1 == comm->M4) {
    for(xx = 0; xx < comm->m1; xx += Px2) {
      int x_end = min(comm->m1, xx + Px2);
      for (zz = from_z; zz < to_z; zz += Pz2) {
        int z_end = min(to_z, zz + Pz2);
        // for each sub-tile
        // FFTy
        int x, y, z;
        for (x = xx; x < x_end; x++) {
          for (z = zz; z < z_end; z++) {
            mpi_t = MPI_Wtime();
            t[PACK2] += mpi_t;
            t[FFTy2] -= mpi_t;
            if (is_ignore_Ry || x % 10 >= Ry) {
              double *ptr = out + 2*M4*p1*(z + M3 * x);
              fftw_execute_dft(po->p1d_y, (fftw_complex*)ptr, (fftw_complex*)ptr);
            }
            pencil_count++;
            mpi_t = MPI_Wtime();
            t[FFTy2] += mpi_t;
            t[TEST2] -= mpi_t;
            if (
              W > 0 &&
              Fy2 > 0 &&
              tile_ind > 0 &&
              test_count < Fy2 &&
              num_pencil * (test_count + 1) / Fy2 == pencil_count)
            {
              communicate_test(buffers[test_curr_tile_ind % (W + 1)]);
              test_curr_tile_ind++;
              if (test_curr_tile_ind > test_max_tile_ind)
                test_curr_tile_ind = test_min_tile_ind;
              test_count++;
            }
            mpi_t = MPI_Wtime();
            t[PACK2] -= mpi_t;
            t[TEST2] += mpi_t;
          }
        } // end of x
#ifdef A2AV
        // pack: xzy => yzxB
        /* loop order: destination yzxB */
        for (y = 0; y < Ny; y++) {
          for (z = zz; z < z_end; z++) {
            for (x = xx; x < x_end; x++) {
              int a, B, y_off, Sz, Sy;
              /* a: target rank index comm->ranks2[] determined by y */
              /* B: begin address in buffer for a reflecting T */
              /* y_off: begin address on y-dim in out for a */
              /* Sy: y-stride in buffer: myT */
              /* Sx: x-stride in buffer: myT * (F4+1) */
              if (is_a2av) {
                if (F4*(p1-b4) <= y) {
                  a = (y - F4*(p1-b4)) / (F4+1) + (p1-b4);
                  B = (p1-b4)*(m1*F4*myT) + (a-(p1-b4))*(m1*(F4+1)*myT);
                  y_off = (p1-b4)*F4 + (a-(p1-b4))*(F4+1);
                  Sz = m1;
                  Sy = m1 * myT;
                } else {
                  a = y / F4;
                  B = a*(m1*F4*myT);
                  y_off = a*F4;
                  Sz = m1;
                  Sy = m1 * myT;
                }
              } else {
                if (F4*(p1-b4) <= y) {
                  a = (y - F4*(p1-b4)) / (F4+1) + (p1-b4);
                  B = a*(M1*M4*myT);
                  y_off = (p1-b4)*F4 + (a-(p1-b4))*(F4+1);
                  Sz = M1;
                  Sy = M1 * myT;
                } else {
                  a = y / F4;
                  B = a*(M1*M4*myT);
                  y_off = a*F4;
                  Sz = M1;
                  Sy = M1 * myT;
                }
              }
              memcpy(
                buffer->a2as + 2*(B + x + (z-from_z)*Sz + (y-y_off)*Sy),
                out + 2*y + 2*Ny*z + 2*M4*p1*M3*x,
                2 * sizeof(double));
            }
          }
        }
#else // A2AV
        // pack: xzy => yzxB
        /* loop order: destination yzxB */
        for (y = 0; y < Ny; y++) {
          for (z = zz; z < z_end; z++) {
            for (x = xx; x < x_end; x++) {
              memcpy(
                buffer->a2as + 2*((y/M4)*(M4*M1*myT) + x + (z-from_z)*M1 + (y%M4)*M1*myT),
                out + 2*(y + (M4*p1)*(z + M3 * x)),
                2 * sizeof(double));
            }
          }
        } /* end of y */
#endif // A2AV
        pack_pencil_count = pencil_count;
        mpi_t = MPI_Wtime();
        t[TEST2] -= mpi_t;
        t[PACK2] += mpi_t;
        if (
          W > 0 &&
          FP2 > 0 &&
          tile_ind > 0 &&
          pack_test_count < FP2 &&
          num_pencil * (pack_test_count + 1) / FP2 == pack_pencil_count)
        {
          communicate_test(buffers[test_curr_tile_ind % (W + 1)]);
          test_curr_tile_ind++;
          if (test_curr_tile_ind > test_max_tile_ind)
            test_curr_tile_ind = test_min_tile_ind;
          pack_test_count++;
        }
        t[PACK2] -= mpi_t;
        t[TEST2] += mpi_t;
      } // end of xx
    } // end of zz
  } else {
    for (zz = from_z; zz < to_z; zz += Pz2) {
      int z_end = min(to_z, zz + Pz2);
      for(xx = 0; xx < comm->m1; xx += Px2) {
        int x_end = min(comm->m1, xx + Px2);
        // for each sub-tile
        // FFTy
        int x, y, z;
        for (z = zz; z < z_end; z++) {
          for (x = xx; x < x_end; x++) {
            mpi_t = MPI_Wtime();
            t[PACK2] += mpi_t;
            t[FFTy2] -= mpi_t;
            if (is_ignore_Ry || x % 10 >= Ry) {
              double *ptr = out + 2*M4*p1*(x + M1 * z);
              fftw_execute_dft(po->p1d_y, (fftw_complex*)ptr, (fftw_complex*)ptr);
            }
            pencil_count++;
            mpi_t = MPI_Wtime();
            t[FFTy2] += mpi_t;
            t[TEST2] -= mpi_t;
            if (
              W > 0 &&
              Fy2 > 0 &&
              tile_ind > 0 &&
              test_count < Fy2 &&
              num_pencil * (test_count + 1) / Fy2 == pencil_count)
            {
              communicate_test(buffers[test_curr_tile_ind % (W + 1)]);
              test_curr_tile_ind++;
              if (test_curr_tile_ind > test_max_tile_ind)
                test_curr_tile_ind = test_min_tile_ind;
              test_count++;
            }
            mpi_t = MPI_Wtime();
            t[PACK2] -= mpi_t;
            t[TEST2] += mpi_t;
          }
        } // end of x
        // pack: zxy => zyxB
        /* loop order: destination zyxB */
#ifdef A2AV
        for (z = zz; z < z_end; z++) {
          for (y = 0; y < Ny; y++) {
            for (x = xx; x < x_end; x++) {
              int a, B, y_off, Sz, Sy;
              /* a: target rank index comm->ranks2[] determined by y */
              /* B: begin address in buffer for a reflecting T */
              /* y_off: begin address on y-dim in out for a */
              /* Sy: y-stride in buffer: myT */
              /* Sx: x-stride in buffer: myT * (F4+1) */
              if (is_a2av) {
                if (F4*(p1-b4) <= y) {
                  a = (y - F4*(p1-b4)) / (F4+1) + (p1-b4);
                  B = (p1-b4)*(m1*F4*myT) + (a-(p1-b4))*(m1*(F4+1)*myT);
                  y_off = (p1-b4)*F4 + (a-(p1-b4))*(F4+1);
                  Sy = m1;
                  Sz = m1 * (F4+1);
                } else {
                  a = y / F4;
                  B = a*(m1*F4*myT);
                  y_off = a*F4;
                  Sy = m1;
                  Sz = m1 * F4;
                }
              } else {
                if (F4*(p1-b4) <= y) {
                  a = (y - F4*(p1-b4)) / (F4+1) + (p1-b4);
                  B = a*(M1*M4*myT);
                  y_off = (p1-b4)*F4 + (a-(p1-b4))*(F4+1);
                  Sy = M1;
                  Sz = M1 * M4;
                } else {
                  a = y / F4;
                  B = a*(M1*M4*myT);
                  y_off = a*F4;
                  Sy = M1;
                  Sz = M1 * M4;
                }
              }
              memcpy(
                buffer->a2as + 2*(B + x + (y-y_off)*Sy + (z-from_z)*Sz),
                out + 2*y + 2*M4*p1*x + 2*M4*p1*M1*z,
                2 * sizeof(double));
            }
          }
        }
#else // A2AV
        for (z = zz; z < z_end; z++) {
          for (y = 0; y < Ny; y++) {
            for (x = xx; x < x_end; x++) { // Mm
              memcpy(
                buffer->a2as + 2*((y/M4)*(M4*M1*myT) + x + (y%M4)*M1 + (z-from_z)*M1*M4),
                out + 2*(y + (M4*p1)*(x + M1 * z)),
                2 * sizeof(double));
            }
          }
        } /* end of z */
#endif // A2AV
        pack_pencil_count = pencil_count;
        mpi_t = MPI_Wtime();
        t[TEST2] -= mpi_t;
        t[PACK2] += mpi_t;
        if (
          W > 0 &&
          FP2 > 0 &&
          tile_ind > 0 &&
          pack_test_count < FP2 &&
          num_pencil * (pack_test_count + 1) / FP2 == pack_pencil_count)
        {
          communicate_test(buffers[test_curr_tile_ind % (W + 1)]);
          test_curr_tile_ind++;
          if (test_curr_tile_ind > test_max_tile_ind)
            test_curr_tile_ind = test_min_tile_ind;
          pack_test_count++;
        }
        mpi_t = MPI_Wtime();
        t[PACK2] -= mpi_t;
        t[TEST2] += mpi_t;
      } // end of xx
    } // end of zz
  } /* end of if is_equalxy */
#endif // STRIDE
  //t[PACK2] += MPI_Wtime();
}

void compute_unpack2_fftx(double *out, struct _offt_plan* po, int tile_ind, int myT)
{
  double *t = po->t;
  double mpi_t = 0.0;
  //t[UNPACK2] -= MPI_Wtime();
  struct _offt_buffer** buffers = (struct _offt_buffer**)(po->buffers2);
  struct _offt_comm *comm = po->comm;
  int T = po->params->v[_T2_];
  int W = po->params->v[_W2_];
  int M3 = comm->M3;
  int M4 = comm->M4;
#ifdef A2AV
  int F1 = comm->F1;
  int m4 = comm->m4;
  int b1 = comm->b1;
  int p1 = comm->p1;
  int is_a2av = po->params->v[_V_] & 1;
#ifdef STRIDE
  int M1 = comm->M1;
#else
  int M1 = comm->M1;
#endif
#else
  int M1 = comm->M1;
#ifdef STRIDE
#else
  int p1 = comm->p1;
#endif
#endif
  int Nx = po->Nx;
  int from_z = tile_ind * T;
  int to_z = from_z + myT;
  struct _offt_buffer* buffer = buffers[tile_ind % (W + 1)];

  int Uz2 = po->params->v[_Uz2_];
  int Uy2 = po->params->v[_Uy2_];
  int FU2 = po->params->v[_FU2_];
  int Fx = po->params->v[_Fx_];
  //int num_pencil = T / Uz2 * comm->m4 / Uy2;
  int num_pencil = T * comm->M4;
  /* in case F values are too big */
  FU2 = min(num_pencil, FU2);
  Fx = min(num_pencil, Fx);
  int unpack_pencil_count = 0;
  int pencil_count = 0;
  int unpack_test_count = 0;
  int test_count = 0;
  int last_tile_ind = (comm->m3 - 1) / T;
  int test_min_tile_ind = min(tile_ind + 1, last_tile_ind);
  int test_max_tile_ind = min(tile_ind + W, last_tile_ind);
  int test_curr_tile_ind = test_min_tile_ind;
  /* loop tiling along z and x dimensions */
  int zz, yy;
#ifdef STRIDE
  if (po->params->v[_S_]) {
    for (yy = 0; yy < comm->m4; yy += Uy2) {
      int y_end = min(comm->m4, yy + Uy2);
      for (zz = from_z; zz < to_z; zz += Uz2) {
        int z_end = min(to_z, zz + Uz2);
        // unpack: xyzB => xyz
        int z, y, x;
#ifdef A2AV
        for (x = 0; x < Nx; x++) {
          for (y = yy; y < y_end; y++) {
            for (z = zz; z < z_end; z++) {
              int a, B, x_off, Sy, Sx;
              /* a: target rank index comm->ranks2[] determined by x */
              /* B: begin address in buffer for a reflecting T */
              /* x_off: begin address on x-dim in out for a */
              /* Sy: y-stride in buffer: myT */
              /* Sx: x-stride in buffer: myT * (F4+1) */
              if (is_a2av) {
                if (F1*(p1-b1) <= x) {
                  a = (x - F1*(p1-b1)) / (F1+1) + (p1-b1);
                  B = (p1-b1)*(F1*m4*myT) + (a-(p1-b1))*((F1+1)*m4*myT);
                  x_off = (p1-b1)*F1 + (a-(p1-b1))*(F1+1);
                  Sy = myT;
                  Sx = myT * m4;
                } else {
                  a = x / F1;
                  B = a*(F1*m4*myT);
                  x_off = a*F1;
                  Sy = myT;
                  Sx = myT * m4;
                }
              } else {
                if (F1*(p1-b1) <= x) {
                  a = (x - F1*(p1-b1)) / (F1+1) + (p1-b1);
                  B = a*(M1*M4*myT);
                  x_off = (p1-b1)*F1 + (a-(p1-b1))*(F1+1);
                  Sy = myT;
                  Sx = myT * M4;
                } else {
                  a = x / F1;
                  B = a*(M1*M4*myT);
                  x_off = a*F1;
                  Sy = myT;
                  Sx = myT * M4;
                }
              }
              memcpy(
                out + 2*z + 2*M3*y + 2*M3*M4*x,
                buffer->a2ar + 2*(B + (z-from_z) + y*Sy + (x-x_off)*Sx),
                2 * sizeof(double));
            }
            unpack_pencil_count++;
          }
        }
#else
        for (x = 0; x < Nx; x++) {
          for (y = yy; y < y_end; y++) {
            //for (z = zz; z < z_end; z++) {
            /* consider xB (M1 complex numbers on x dim) as one element. */
            memcpy(
              out + 2*(zz + M3 * y + M3*M4 * x),
              buffer->a2ar + 2*((x/M1)*(M4*M1*myT) + (zz-from_z) + y*myT + M4*myT*(x%M1)),
              2 * (z_end - zz) * sizeof(double));
            //}
            unpack_pencil_count++;
          }
        }
#endif
        mpi_t = MPI_Wtime();
        t[TEST2] -= mpi_t;
        t[UNPACK2] += mpi_t;
        if (
          W > 0 &&
          FU2 > 0 &&
          unpack_test_count < FU2 &&
          num_pencil * (unpack_test_count + 1) / FU2 == unpack_pencil_count)
        {
          communicate_test(buffers[test_curr_tile_ind % (W + 1)]);
          test_curr_tile_ind++;
          if (test_curr_tile_ind > test_max_tile_ind)
            test_curr_tile_ind = test_min_tile_ind;
          unpack_test_count++;
        }
        mpi_t = MPI_Wtime();
        t[UNPACK2] -= mpi_t;
        t[TEST2] += mpi_t;
        // FFTx
        for (y = yy; y < y_end; y++)
          for (z = zz; z < z_end; z++) {
            mpi_t = MPI_Wtime();
            t[UNPACK2] += mpi_t;
            t[FFTx] -= mpi_t;
            double *ptr = out + 2*z + 2* M3 * y;
            //double *ptr = out + 2*z + 2*(M1*p1)*(M3 * y);
            fftw_execute_dft(po->p1d_x, (fftw_complex*)ptr, (fftw_complex*)ptr);
            pencil_count++;
            mpi_t = MPI_Wtime();
            t[FFTx] += mpi_t;
            t[TEST2] -= mpi_t;
            if (
              W > 0 &&
              Fx > 0 &&
              test_count < Fx &&
              num_pencil * (test_count + 1) / Fx == pencil_count)
            {
              communicate_test(buffers[test_curr_tile_ind % (W + 1)]);
              test_curr_tile_ind++;
              if (test_curr_tile_ind > test_max_tile_ind)
                test_curr_tile_ind = test_min_tile_ind;
              test_count++;
            }
            mpi_t = MPI_Wtime();
            t[UNPACK2] -= mpi_t;
            t[TEST2] += mpi_t;
          } /* end of for z */
      } // end of yy
    } // end of zz
  } else {
    if (po->is_equalxy && comm->M1 == comm->M4) {
      /* yzxB->yzx when Nx == Ny */
      for (yy = 0; yy < comm->m4; yy += Uy2) {
        int y_end = min(comm->m4, yy + Uy2);
        for (zz = from_z; zz < to_z; zz += Uz2) {
          int z_end = min(to_z, zz + Uz2);
          // unpack: yzxB => yzx
          int z, y, x;
          /* loop order: destination yzx */
#ifdef A2AV
        for (y = yy; y < y_end; y++) {
          for (z = zz; z < z_end; z++) {
            for (x = 0; x < Nx; x++) {
              int a, B, x_off, Sz, Sy;
              /* a: target rank index comm->ranks2[] determined by x */
              /* B: begin address in buffer for a reflecting T */
              /* x_off: begin address on x-dim in out for a */
              /* Sy: y-stride in buffer: myT */
              /* Sx: x-stride in buffer: myT * (F4+1) */
              if (is_a2av) {
                if (F1*(p1-b1) <= x) {
                  a = (x - F1*(p1-b1)) / (F1+1) + (p1-b1);
                  B = (p1-b1)*(F1*m4*myT) + (a-(p1-b1))*((F1+1)*m4*myT);
                  x_off = (p1-b1)*F1 + (a-(p1-b1))*(F1+1);
                  Sz = (F1+1);
                  Sy = (F1+1) * myT;
                } else {
                  a = x / F1;
                  B = a*(F1*m4*myT);
                  x_off = a*F1;
                  Sz = F1;
                  Sy = F1 * myT;
                }
              } else {
                if (F1*(p1-b1) <= x) {
                  a = (x - F1*(p1-b1)) / (F1+1) + (p1-b1);
                  B = a*(M1*M4*myT);
                  x_off = (p1-b1)*F1 + (a-(p1-b1))*(F1+1);
                  Sz = M1;
                  Sy = M1 * myT;
                } else {
                  a = x / F1;
                  B = a*(M1*M4*myT);
                  x_off = a*F1;
                  Sz = M1;
                  Sy = M1 * myT;
                }
              }
              memcpy(
                out + 2*(x + M1*p1*(z + M3 * y)),
                buffer->a2ar + 2*(B + (x-x_off) + (z-from_z)*Sz + y*Sy),
                2 * sizeof(double));
            }
            unpack_pencil_count++;
          }
        }
#else
          for (y = yy; y < y_end; y++)
            for (z = zz; z < z_end; z++) {
              for (x = 0; x < Nx; x += M1) {
                /* consider xB (M1 complex numbers on x dim) as one element. */
                memcpy(
                  out + 2*(x + (M1*p1)*(z + M3 * y)),
                  buffer->a2ar + 2*((x/M1)*(M4*M1*myT) + (x%M1) + M1*(z-from_z) + M1*myT*y),
                  2 * min(M1, Nx-x) * sizeof(double));
              }
              unpack_pencil_count++;
            }
#endif
          mpi_t = MPI_Wtime();
          t[TEST2] -= mpi_t;
          t[UNPACK2] += mpi_t;
          if (
            W > 0 &&
            FU2 > 0 &&
            unpack_test_count < FU2 &&
            num_pencil * (unpack_test_count + 1) / FU2 == unpack_pencil_count)
          {
            communicate_test(buffers[test_curr_tile_ind % (W + 1)]);
            test_curr_tile_ind++;
            if (test_curr_tile_ind > test_max_tile_ind)
              test_curr_tile_ind = test_min_tile_ind;
            unpack_test_count++;
          }
          mpi_t = MPI_Wtime();
          t[UNPACK2] -= mpi_t;
          t[TEST2] += mpi_t;
          // FFTx
          for (y = yy; y < y_end; y++)
            for (z = zz; z < z_end; z++) {
              mpi_t = MPI_Wtime();
              t[UNPACK2] += mpi_t;
              t[FFTx] -= mpi_t;
              double *ptr = out + 2*(M1*p1)*(z + M3 * y);
              fftw_execute_dft(po->p1d_x, (fftw_complex*)ptr, (fftw_complex*)ptr);
              pencil_count++;
              mpi_t = MPI_Wtime();
              t[FFTx] += mpi_t;
              t[TEST2] -= mpi_t;
              if (
                W > 0 &&
                Fx > 0 &&
                test_count < Fx &&
                num_pencil * (test_count + 1) / Fx == pencil_count)
              {
                communicate_test(buffers[test_curr_tile_ind % (W + 1)]);
                test_curr_tile_ind++;
                if (test_curr_tile_ind > test_max_tile_ind)
                  test_curr_tile_ind = test_min_tile_ind;
                test_count++;
              }
              mpi_t = MPI_Wtime();
              t[UNPACK2] -= mpi_t;
              t[TEST2] += mpi_t;
            } /* end of for z */
        } // end of yy
      } // end of zz
    } else {
      /* zyxB->zyx */
      for (zz = from_z; zz < to_z; zz += Uz2) {
        int z_end = min(to_z, zz + Uz2);
        for (yy = 0; yy < comm->m4; yy += Uy2) {
          int y_end = min(comm->m4, yy + Uy2);
          // unpack: zyxB => zyx
          int z, y, x;
          /* loop order: destination zyx */
#ifdef A2AV
          for (z = zz; z < z_end; z++) {
            for (y = yy; y < y_end; y++) {
              for (x = 0; x < Nx; x++) {
                int a, B, x_off, Sz, Sy;
                /* a: target rank index comm->ranks2[] determined by x */
                /* B: begin address in buffer for a reflecting T */
                /* x_off: begin address on x-dim in out for a */
                /* Sy: y-stride in buffer: myT */
                /* Sx: x-stride in buffer: myT * (F4+1) */
                if (is_a2av) {
                  if (F1*(p1-b1) <= x) {
                    a = (x - F1*(p1-b1)) / (F1+1) + (p1-b1);
                    B = (p1-b1)*(F1*m4*myT) + (a-(p1-b1))*((F1+1)*m4*myT);
                    x_off = (p1-b1)*F1 + (a-(p1-b1))*(F1+1);
                    Sy = (F1+1);
                    Sz = (F1+1) * m4;
                  } else {
                    a = x / F1;
                    B = a*(F1*m4*myT);
                    x_off = a*F1;
                    Sy = F1;
                    Sz = F1 * m4;
                  }
                } else {
                  if (F1*(p1-b1) <= x) {
                    a = (x - F1*(p1-b1)) / (F1+1) + (p1-b1);
                    B = a*(M1*M4*myT);
                    x_off = (p1-b1)*F1 + (a-(p1-b1))*(F1+1);
                    Sy = M1;
                    Sz = M1 * M4;
                  } else {
                    a = x / F1;
                    B = a*(M1*M4*myT);
                    x_off = a*F1;
                    Sy = M1;
                    Sz = M1 * M4;
                  }
                }
                memcpy(
                  out + 2*(x + M1*p1*(y + M4 * z)),
                  buffer->a2ar + 2*(B + (x-x_off) + y*Sy + (z-from_z)*Sz),
                  2 * sizeof(double));
              }
              unpack_pencil_count++;
            }
          }
#else
          for (z = zz; z < z_end; z++)
            for (y = yy; y < y_end; y++) {
              for (x = 0; x < Nx; x+=M1) {
                /* consider xB (M1 complex numbers on x dim) as one element. */
                memcpy(
                  out + 2*(x + (M1*p1)*(y + M4 * z)),
                  buffer->a2ar + 2*((x/M1)*(M4*M1*myT) + (x%M1) + y*M1 + M1*M4*(z-from_z)),
                  2 * min(M1, Nx-x) * sizeof(double));
              }
              unpack_pencil_count++;
            }
#endif
          mpi_t = MPI_Wtime();
          t[TEST2] -= mpi_t;
          t[UNPACK2] += mpi_t;
          if (
            W > 0 &&
            FU2 > 0 &&
            unpack_test_count < FU2 &&
            num_pencil * (unpack_test_count + 1) / FU2 == unpack_pencil_count)
          {
            communicate_test(buffers[test_curr_tile_ind % (W + 1)]);
            test_curr_tile_ind++;
            if (test_curr_tile_ind > test_max_tile_ind)
              test_curr_tile_ind = test_min_tile_ind;
            unpack_test_count++;
          }
          mpi_t = MPI_Wtime();
          t[UNPACK2] -= mpi_t;
          t[TEST2] += mpi_t;
          // FFTx
          for (z = zz; z < z_end; z++)
            for (y = yy; y < y_end; y++) { // Mm
              mpi_t = MPI_Wtime();
              t[UNPACK2] += mpi_t;
              t[FFTx] -= mpi_t;
              double *ptr = out + 2*(M1*p1)*(y + M4 * z);
              fftw_execute_dft(po->p1d_x, (fftw_complex*)ptr, (fftw_complex*)ptr);
              pencil_count++;
              mpi_t = MPI_Wtime();
              t[FFTx] += mpi_t;
              t[TEST2] -= mpi_t;
              if (
                W > 0 &&
                Fx > 0 &&
                test_count < Fx &&
                num_pencil * (test_count + 1) / Fx == pencil_count)
              {
                communicate_test(buffers[test_curr_tile_ind % (W + 1)]);
                test_curr_tile_ind++;
                if (test_curr_tile_ind > test_max_tile_ind)
                  test_curr_tile_ind = test_min_tile_ind;
                test_count++;
              }
              mpi_t = MPI_Wtime();
              t[UNPACK2] -= mpi_t;
              t[TEST2] += mpi_t;
            }
        } // end of yy
      } // end of zz
    } /* end of if is_equalxy */
  } /* end of if stride */
#else // STRIDE
  if (po->is_equalxy && comm->M1 == comm->M4) {
    /* yzxB->yzx when Nx == Ny */
    for (yy = 0; yy < comm->m4; yy += Uy2) {
      int y_end = min(comm->m4, yy + Uy2);
      for (zz = from_z; zz < to_z; zz += Uz2) {
        int z_end = min(to_z, zz + Uz2);
        // unpack: yzxB => yzx
        int z, y, x;
        /* loop order: destination yzx */
#ifdef A2AV
      for (y = yy; y < y_end; y++) {
        for (z = zz; z < z_end; z++) {
          for (x = 0; x < Nx; x++) {
            int a, B, x_off, Sz, Sy;
            /* a: target rank index comm->ranks2[] determined by x */
            /* B: begin address in buffer for a reflecting T */
            /* x_off: begin address on x-dim in out for a */
            /* Sy: y-stride in buffer: myT */
            /* Sx: x-stride in buffer: myT * (F4+1) */
            if (is_a2av) {
              if (F1*(p1-b1) <= x) {
                a = (x - F1*(p1-b1)) / (F1+1) + (p1-b1);
                B = (p1-b1)*(F1*m4*myT) + (a-(p1-b1))*((F1+1)*m4*myT);
                x_off = (p1-b1)*F1 + (a-(p1-b1))*(F1+1);
                Sz = (F1+1);
                Sy = (F1+1) * myT;
              } else {
                a = x / F1;
                B = a*(F1*m4*myT);
                x_off = a*F1;
                Sz = F1;
                Sy = F1 * myT;
              }
            } else {
              if (F1*(p1-b1) <= x) {
                a = (x - F1*(p1-b1)) / (F1+1) + (p1-b1);
                B = a*(M1*M4*myT);
                x_off = (p1-b1)*F1 + (a-(p1-b1))*(F1+1);
                Sz = M1;
                Sy = M1 * myT;
              } else {
                a = x / F1;
                B = a*(M1*M4*myT);
                x_off = a*F1;
                Sz = M1;
                Sy = M1 * myT;
              }
            }
            memcpy(
              out + 2*(x + M1*p1*(z + M3 * y)),
              buffer->a2ar + 2*(B + (x-x_off) + (z-from_z)*Sz + y*Sy),
              2 * sizeof(double));
          }
          unpack_pencil_count++;
        }
      }
#else
        for (y = yy; y < y_end; y++)
          for (z = zz; z < z_end; z++) {
            for (x = 0; x < Nx; x += M1) {
              /* consider xB (M1 complex numbers on x dim) as one element. */
              memcpy(
                out + 2*(x + (M1*p1)*(z + M3 * y)),
                buffer->a2ar + 2*((x/M1)*(M4*M1*myT) + (x%M1) + M1*(z-from_z) + M1*myT*y),
                2 * min(M1, Nx-x) * sizeof(double));
            }
            unpack_pencil_count++;
          }
#endif
        mpi_t = MPI_Wtime();
        t[TEST2] -= mpi_t;
        t[UNPACK2] += mpi_t;
        if (
          W > 0 &&
          FU2 > 0 &&
          unpack_test_count < FU2 &&
          num_pencil * (unpack_test_count + 1) / FU2 == unpack_pencil_count)
        {
          communicate_test(buffers[test_curr_tile_ind % (W + 1)]);
          test_curr_tile_ind++;
          if (test_curr_tile_ind > test_max_tile_ind)
            test_curr_tile_ind = test_min_tile_ind;
          unpack_test_count++;
        }
        mpi_t = MPI_Wtime();
        t[UNPACK2] -= mpi_t;
        t[TEST2] += mpi_t;
        // FFTx
        for (y = yy; y < y_end; y++)
          for (z = zz; z < z_end; z++) {
            mpi_t = MPI_Wtime();
            t[UNPACK2] += mpi_t;
            t[FFTx] -= mpi_t;
            double *ptr = out + 2*(M1*p1)*(z + M3 * y);
            fftw_execute_dft(po->p1d_x, (fftw_complex*)ptr, (fftw_complex*)ptr);
            pencil_count++;
            mpi_t = MPI_Wtime();
            t[FFTx] += mpi_t;
            t[TEST2] -= mpi_t;
            if (
              W > 0 &&
              Fx > 0 &&
              test_count < Fx &&
              num_pencil * (test_count + 1) / Fx == pencil_count)
            {
              communicate_test(buffers[test_curr_tile_ind % (W + 1)]);
              test_curr_tile_ind++;
              if (test_curr_tile_ind > test_max_tile_ind)
                test_curr_tile_ind = test_min_tile_ind;
              test_count++;
            }
            mpi_t = MPI_Wtime();
            t[UNPACK2] -= mpi_t;
            t[TEST2] += mpi_t;
          } /* end of for z */
      } // end of yy
    } // end of zz
  } else {
    /* zyxB->zyx */
    for (zz = from_z; zz < to_z; zz += Uz2) {
      int z_end = min(to_z, zz + Uz2);
      for (yy = 0; yy < comm->m4; yy += Uy2) {
        int y_end = min(comm->m4, yy + Uy2);
        // unpack: zyxB => zyx
        int z, y, x;
        /* loop order: destination zyx */
#ifdef A2AV
        for (z = zz; z < z_end; z++) {
          for (y = yy; y < y_end; y++) {
            for (x = 0; x < Nx; x++) {
              int a, B, x_off, Sz, Sy;
              /* a: target rank index comm->ranks2[] determined by x */
              /* B: begin address in buffer for a reflecting T */
              /* x_off: begin address on x-dim in out for a */
              /* Sy: y-stride in buffer: myT */
              /* Sx: x-stride in buffer: myT * (F4+1) */
              if (is_a2av) {
                if (F1*(p1-b1) <= x) {
                  a = (x - F1*(p1-b1)) / (F1+1) + (p1-b1);
                  B = (p1-b1)*(F1*m4*myT) + (a-(p1-b1))*((F1+1)*m4*myT);
                  x_off = (p1-b1)*F1 + (a-(p1-b1))*(F1+1);
                  Sy = (F1+1);
                  Sz = (F1+1) * m4;
                } else {
                  a = x / F1;
                  B = a*(F1*m4*myT);
                  x_off = a*F1;
                  Sy = F1;
                  Sz = F1 * m4;
                }
              } else {
                if (F1*(p1-b1) <= x) {
                  a = (x - F1*(p1-b1)) / (F1+1) + (p1-b1);
                  B = a*(M1*M4*myT);
                  x_off = (p1-b1)*F1 + (a-(p1-b1))*(F1+1);
                  Sy = M1;
                  Sz = M1 * M4;
                } else {
                  a = x / F1;
                  B = a*(M1*M4*myT);
                  x_off = a*F1;
                  Sy = M1;
                  Sz = M1 * M4;
                }
              }
              memcpy(
                out + 2*(x + M1*p1*(y + M4 * z)),
                buffer->a2ar + 2*(B + (x-x_off) + y*Sy + (z-from_z)*Sz),
                2 * sizeof(double));
            }
            unpack_pencil_count++;
          }
        }
#else
        for (z = zz; z < z_end; z++)
          for (y = yy; y < y_end; y++) {
            for (x = 0; x < Nx; x+=M1) {
              /* consider xB (M1 complex numbers on x dim) as one element. */
              memcpy(
                out + 2*(x + (M1*p1)*(y + M4 * z)),
                buffer->a2ar + 2*((x/M1)*(M4*M1*myT) + (x%M1) + y*M1 + M1*M4*(z-from_z)),
                2 * min(M1, Nx-x) * sizeof(double));
            }
            unpack_pencil_count++;
          }
#endif
        mpi_t = MPI_Wtime();
        t[TEST2] -= mpi_t;
        t[UNPACK2] += mpi_t;
        if (
          W > 0 &&
          FU2 > 0 &&
          unpack_test_count < FU2 &&
          num_pencil * (unpack_test_count + 1) / FU2 == unpack_pencil_count)
        {
          communicate_test(buffers[test_curr_tile_ind % (W + 1)]);
          test_curr_tile_ind++;
          if (test_curr_tile_ind > test_max_tile_ind)
            test_curr_tile_ind = test_min_tile_ind;
          unpack_test_count++;
        }
        mpi_t = MPI_Wtime();
        t[UNPACK2] -= mpi_t;
        t[TEST2] += mpi_t;
        // FFTx
        for (z = zz; z < z_end; z++)
          for (y = yy; y < y_end; y++) { // Mm
            mpi_t = MPI_Wtime();
            t[UNPACK2] += mpi_t;
            t[FFTx] -= mpi_t;
            double *ptr = out + 2*(M1*p1)*(y + M4 * z);
            fftw_execute_dft(po->p1d_x, (fftw_complex*)ptr, (fftw_complex*)ptr);
            pencil_count++;
            mpi_t = MPI_Wtime();
            t[FFTx] += mpi_t;
            t[TEST2] -= mpi_t;
            if (
              W > 0 &&
              Fx > 0 &&
              test_count < Fx &&
              num_pencil * (test_count + 1) / Fx == pencil_count)
            {
              communicate_test(buffers[test_curr_tile_ind % (W + 1)]);
              test_curr_tile_ind++;
              if (test_curr_tile_ind > test_max_tile_ind)
                test_curr_tile_ind = test_min_tile_ind;
              test_count++;
            }
            mpi_t = MPI_Wtime();
            t[UNPACK2] -= mpi_t;
            t[TEST2] += mpi_t;
          }
      } // end of yy
    } // end of zz
  } /* end of if is_equalxy */
#endif // STRIDE
  //t[UNPACK2] += MPI_Wtime();
}

/* **********************************************************
 @ setup "struct _offt_params"
   ********************************************************** */
void params_range_setup(struct _offt_plan *po, int** v_list, int* v_list_size) {
  int c;
  int i;
  for (i = 0; i < PARAM_COUNT; i++) {
    if (i == _P1_) {
      /* P1: divisors of p */
      /* count first */
      int p = po->p;
      c = 0;
      int pi1;
      int Nz_new = (po->is_r2c)?po->Nz/2+1:po->Nz;
      /*int sp = (int)sqrt(p);*/
      int p_u = min( min(po->Nx,po->Ny), p );
      int p_l = max( max(p/Nz_new,p/po->Ny), 1 );
      for (pi1 = p_l; pi1 <= p_u; pi1++) {
        if (p % pi1 != 0) continue;
        c++;
      }
      v_list_size[_P1_] = c;
      /* fill in the list */
      v_list[_P1_] = (int*)malloc(sizeof(int) * v_list_size[_P1_]);
      c = 0;
      for (pi1 = p_l; pi1 <= p_u; pi1++) {
        if (p % pi1 != 0) continue;
        v_list[_P1_][c] = pi1; c++;
      }
    } else if (i == _W1_ || i == _W2_ || i == _Ry_) {
      v_list_size[i] = 11;
      v_list[i] = (int*)malloc(sizeof(int) * v_list_size[i]);
      for (c = 0; c < v_list_size[i]; c++) {
        v_list[i][c] = c;
      }
    } else if (i == _V_) {
      v_list_size[i] = 4;
      v_list[i] = (int*)malloc(sizeof(int) * v_list_size[i]);
      for (c = 0; c < v_list_size[i]; c++) {
        v_list[i][c] = c;
      }
    } else if (i == _S_) {
      v_list_size[i] = 2;
      v_list[i] = (int*)malloc(sizeof(int) * v_list_size[i]);
      for (c = 0; c < v_list_size[i]; c++) {
        v_list[i][c] = c;
      }
    } else {
      int Nx = po->Nx;
      int Ny = po->Ny;
      int Nz_new = (po->is_r2c)?po->Nz/2+1:po->Nz;
      int v_max = -1;
      int is_put_zero = 0;
      switch (i) {
      case _T1_: v_max = Nx; is_put_zero = 0; break;
      case _Px1_: v_max = Nx; is_put_zero = 0; break;
      case _Py1_: v_max = Ny; is_put_zero = 0; break;
      case _Ux1_: v_max = Nx; is_put_zero = 0; break;
      case _Uz1_: v_max = Nz_new; is_put_zero = 0; break;
      case _T2_: v_max = Nz_new; is_put_zero = 0; break;
      case _Px2_: v_max = Nx; is_put_zero = 0; break;
      case _Pz2_: v_max = Nz_new; is_put_zero = 0; break;
      case _Uy2_: v_max = Ny; is_put_zero = 0; break;
      case _Uz2_: v_max = Nz_new; is_put_zero = 0; break;
      case _Fz_: v_max = Nx*Ny; is_put_zero = 1; break; /* add 0 at the beginning */
      case _FP1_: v_max = Nx*Ny; is_put_zero = 1; break;
      case _Fy1_: v_max = Nx*Nz_new; is_put_zero = 1; break;
      case _FU1_: v_max = Nx*Nz_new; is_put_zero = 1; break;
      case _Fy2_: v_max = Nx*Nz_new; is_put_zero = 1; break;
      case _FP2_: v_max = Nx*Nz_new; is_put_zero = 1; break;
      case _FU2_: v_max = Ny*Nz_new; is_put_zero = 1; break;
      case _Fx_: v_max = Ny*Nz_new; is_put_zero = 1; break;
      }
      int l_v_max = floor_log(v_max);
      v_list_size[i] = l_v_max + 1;
      int is_put_max = (inv_log(l_v_max) < v_max)?1:0;
      if (is_put_max) v_list_size[i]++;
      if (is_put_zero) v_list_size[i]++;
      v_list[i] = (int*)malloc(sizeof(int) * v_list_size[i]);
      c = 0;
      if (is_put_zero) { v_list[i][c] = 0; c++; }
      int cc = 0;
      for (cc = 0; cc < l_v_max + 1; cc++) { v_list[i][c] = inv_log(cc); c++; }
      if (is_put_max) { v_list[i][c] = v_max; c++; }
    }
#if 0
    if (!po->rank) {
      printf("%s: param %d: ", __FUNCTION__, i);
      for (c = 0; c < v_list_size[i]; c++) {
        printf("%d ", v_list[i][c]);
      }
      printf("\n");
    }
#endif
  } /* end of for i */

  //*p_v_list_size = v_list_size;
  //return v_list;
}

/* return grid_value 2^floor(log(raw_v)) */
int grid_value_floor(int is_index, int **v_list, int *v_list_size, int i, int raw_v) {
  int grid_v = raw_v;
  /* i: param index */
  int j; /* value index for param i */
  for (j = v_list_size[i] - 1; j >= 0; j--) {
    if (v_list[i][j] <= raw_v) {
      if (is_index) grid_v = j;
      else grid_v = v_list[i][j];
      break;
    }
  }
  /*printf("param %d: %d -> %d\n", i, raw_v, grid_v);*/
  return grid_v;
}

int grid_value_ceil(int is_index, int **v_list, int *v_list_size, int i, int raw_v) {
  int grid_v = raw_v;
  /* i: param index */
  int j; /* value index for param i */
  for (j = 0; j <= v_list_size[i] - 1; j++) {
    //printf("v_list[%d][%d] = %d\n", i, j, v_list[i][j]);
    if (v_list[i][j] >= raw_v) {
      if (is_index) grid_v = j;
      else grid_v = v_list[i][j];
      break;
    }
  }
  /*printf("param %d: %d -> %d\n", i, raw_v, grid_v);*/
  return grid_v;
}

void params_set_default(struct _offt_plan *po) {
  int *v_list[PARAM_COUNT], v_list_size[PARAM_COUNT];
  params_range_setup(po, v_list, v_list_size); /* TODO: avoid a redundant call in ah_tuning */
  int i;

  po->params->is_converged = 1;
  po->params->is_infeasible = 0;
  int *v = po->params->v;

  /* set parameters */
  int p = po->p;
  v[_P1_] = (int)sqrt(p);
  i = _P1_; v[i] = grid_value_floor(0, v_list, v_list_size, i, v[i]);
  int p2 = p / v[_P1_];
  int Nz_new = (po->is_r2c)?(po->Nz/2+1):po->Nz;
  int M1 = (po->Nx + v[_P1_] - 1) / v[_P1_];
  int M2 = (po->Ny + p2 - 1) / p2;
  int M3 = (Nz_new + p2 - 1) / p2;
  int M4 = (po->Ny + v[_P1_] - 1) / v[_P1_];

  /* phase 1 */
  v[_T1_] = max(M1/16, 1); /* some degree of overlap */
  i = _T1_; v[i] = grid_value_floor(0, v_list, v_list_size, i, v[i]);
  v[_W1_] = min(max(2, 0), (M1+v[_T1_]-1)/v[_T1_]);
  i = _W1_; v[i] = grid_value_floor(0, v_list, v_list_size, i, v[i]);
  int P1_xy = 8192/Nz_new; /* 256KB cache: 8192 complex numbers for each read/write */
  v[_Px1_] = min(max((int)sqrt(P1_xy), 1), v[_T1_]);
  i = _Px1_; v[i] = grid_value_floor(0, v_list, v_list_size, i, v[i]);
  v[_Py1_] = min(max(P1_xy/v[_Px1_], 1), M2);
  i = _Py1_; v[i] = grid_value_floor(0, v_list, v_list_size, i, v[i]);
  v[_Fz_] = min(max(p2/2, 0), v[_T1_]*M2);
  i = _Fz_; v[i] = grid_value_floor(0, v_list, v_list_size, i, v[i]);
  v[_FP1_] = min(max(v[_Fz_], 0), v[_T1_]/v[_Px1_]*M2/v[_Py1_]);
  i = _FP1_; v[i] = grid_value_floor(0, v_list, v_list_size, i, v[i]);
  int U1_xz = 8192/po->Ny;
  v[_Ux1_] = min(max((int)sqrt(U1_xz), 1), v[_T1_]);
  i = _Ux1_; v[i] = grid_value_floor(0, v_list, v_list_size, i, v[i]);
  v[_Uz1_] = min(max(U1_xz/v[_Ux1_], 1), M3);
  i = _Uz1_; v[i] = grid_value_floor(0, v_list, v_list_size, i, v[i]);
  v[_FU1_] = min(max(v[_Fz_], 0), v[_T1_]/v[_Ux1_]*M3/v[_Uz1_]);
  i = _FU1_; v[i] = grid_value_floor(0, v_list, v_list_size, i, v[i]);
  v[_Fy1_] = min(max(v[_Fz_], 0), v[_T1_]*M3);
  i = _Fy1_; v[i] = grid_value_floor(0, v_list, v_list_size, i, v[i]);
  v[_Ry_] = 5; /* half */
  /* phase 2 */
  v[_T2_] = max(M3/16, 1);
  i = _T2_; v[i] = grid_value_floor(0, v_list, v_list_size, i, v[i]);
  v[_W2_] = min(max(2, 0), (M3+v[_T2_]-1)/v[_T2_]);
  i = _W2_; v[i] = grid_value_floor(0, v_list, v_list_size, i, v[i]);
  /* follow zxy order of source */
  int P2_xz = 8192/po->Ny;
  v[_Pz2_] = min(max((int)sqrt(P2_xz), 1), v[_T2_]);
  i = _Pz2_; v[i] = grid_value_floor(0, v_list, v_list_size, i, v[i]);
  v[_Px2_] = min(max(P2_xz/v[_Pz2_], 1), M1);
  i = _Px2_; v[i] = grid_value_floor(0, v_list, v_list_size, i, v[i]);
  v[_Fy2_] = min(max(v[_P1_]/2, 0), v[_T2_]*M1);
  i = _Fy2_; v[i] = grid_value_floor(0, v_list, v_list_size, i, v[i]);
  v[_FP2_] = min(max(v[_Fy2_], 0), M1/v[_Px2_]*v[_T2_]/v[_Pz2_]);
  i = _FP2_; v[i] = grid_value_floor(0, v_list, v_list_size, i, v[i]);
  /* follow zyx order of source */
  int U2_yz = 8192/po->Nx;
  v[_Uz2_] = min(max((int)sqrt(U2_yz), 1), v[_T2_]);
  i = _Uz2_; v[i] = grid_value_floor(0, v_list, v_list_size, i, v[i]);
  v[_Uy2_] = min(max(U2_yz/v[_Uz2_], 1), M4);
  i = _Uy2_; v[i] = grid_value_floor(0, v_list, v_list_size, i, v[i]);
  v[_FU2_] = min(max(v[_FP2_], 0), M4/v[_Uy2_]*v[_T2_]/v[_Uz2_]);
  i = _FU2_; v[i] = grid_value_floor(0, v_list, v_list_size, i, v[i]);
  v[_Fx_] = min(max(v[_FP2_], 0), v[_T2_]*M4);
  i = _Fx_; v[i] = grid_value_floor(0, v_list, v_list_size, i, v[i]);
#if 0
  /* special 1xp case */
if (po->is_oned && v[_P1_] == p) {
  v[_Fy_] = min(max(v[_P1_]/2, 0), v[_T2_]*M1);
  i = _Fy_; v[i] = grid_value_floor(0, v_list, v_list_size, i, v[i]);
  int P2_xz = 8192/po->Ny; /* 256KB cache: 8192 complex numbers for each read/write */
  v[_Px2_] = min(max((int)sqrt(P2_xz), 1), M1);
  i = _Px2_; v[i] = grid_value_floor(0, v_list, v_list_size, i, v[i]);
  v[_Py2_] = M2; /* not used */
  i = _Py2_; v[i] = grid_value_floor(0, v_list, v_list_size, i, v[i]);
  v[_Pz2_] = min(max(P2_xz/v[_Px2_], 1), v[_T2_]);
  i = _Pz2_; v[i] = grid_value_floor(0, v_list, v_list_size, i, v[i]);
  v[_FP2_] = min(max(v[_Fy_], 0), M1/v[_Px2_]*v[_T2_]/v[_Pz2_]);
  i = _FP2_; v[i] = grid_value_floor(0, v_list, v_list_size, i, v[i]);
}
#endif
  v[_V_] = 0;
  v[_S_] = 0;
  if (po->is_W0) {
    v[_W1_] = v[_W2_] = 0; 
    v[_Fz_] = v[_FP1_] = v[_FU1_] = v[_Fy1_] = v[_Fy2_] = v[_FP2_] = v[_FU2_] = v[_Fx_] = 0;
  }
#ifdef NOTEST
  if (po->is_notest) { v[_Fz_] = v[_FP1_] = v[_FU1_] = v[_Fy1_] = v[_Fy2_] = v[_FP2_] = v[_FU2_] = v[_Fx_] = 0; }
#endif

  /* cleanup */
  for (i = 0; i < PARAM_COUNT; i++)
    free(v_list[i]);
}

void set_params_custom(struct _offt_plan *po, struct _offt_params *custom_params) {
  int i;
  if (custom_params == NULL) return;
  for (i = 0; i < PARAM_COUNT; i++) {
    if (custom_params->v[i] >= 0)
      po->params->v[i] = custom_params->v[i];
  }
}

/* **********************************************************
 @ print messages
   ********************************************************** */
void print_params(int *v) {
  int i;
  for (i = 0; i < PARAM_COUNT; i++) {
    if (v[i] < 0) continue;
    switch (i) {
    case _P1_: printf("P1 "); break;
    case _T1_: printf("T1 "); break;
    case _W1_: printf("W1 "); break;
    case _Px1_: printf("Px1 "); break;
    case _Py1_: printf("Py1 "); break;
    case _Fz_: printf("Fz "); break;
    case _FP1_: printf("FP1 "); break;
    case _Ux1_: printf("Ux1 "); break;
    case _Uz1_: printf("Uz1 "); break;
    case _Fy1_: printf("Fy1 "); break;
    case _FU1_: printf("FU1 "); break;
    case _T2_: printf("T2 "); break;
    case _W2_: printf("W2 "); break;
    case _Px2_: printf("Px2 "); break;
    case _Ry_: printf("Ry "); break;
    case _Pz2_: printf("Pz2 "); break;
    case _Fy2_: printf("Fy2 "); break;
    case _FP2_: printf("FP2 "); break;
    case _Uy2_: printf("Uy2 "); break;
    case _Uz2_: printf("Uz2 "); break;
    case _FU2_: printf("FU2 "); break;
    case _Fx_: printf("Fx "); break;
    case _V_: printf("V "); break;
    case _S_: printf("S "); break;
    }
    printf("%d ", v[i]);
  }
  printf("\n");
}

#if 0
void print_complex(double *a, int n, int stride) {
  for (int i = 0; i < n*stride; i+=stride) {
    printf("%5.2f:%5.2f ", a[i*2+0], a[i*2+1]);
  }
  printf("\n");
}
#endif

void offt_print_time(double *t) {
  //printf("%.5f  %.3f %.3f %.3f %.3f %.3f %.3f  %.3f  %.3f %.3f %.3f %.3f  %.3f %.3f %.3f %.3f\n",
  printf("%.5f  %.5f %.5f %.5f %.5f %.5f %.5f  %.5f  %.5f %.5f %.5f %.5f  %.5f %.5f %.5f %.5f\n",
    t[ALL], 
    t[INIT1], t[WAIT1], t[TEST1],
    t[INIT2], t[WAIT2], t[TEST2],
    t[TRANSPOSE],
    t[PACK1], t[UNPACK1],
    t[PACK2], t[UNPACK2],
    t[FFTz], t[FFTy1], t[FFTy2], t[FFTx]
  );
}

/* **********************************************************
 @ API functions
   ********************************************************** */
#ifdef NOTEST
struct _offt_plan*
  offt_3d_init(int Nx, int Ny, int Nz, double* in, double* out, int is_r2c, int fftw_flag, int is_oned, int is_a2a, int is_equalxy, int is_notest, int ah_strategy, int max_loop, int tuning_mode, int is_W0, int extrapolation_window, struct _offt_params *custom_params)
#else
struct _offt_plan*
  offt_3d_init(int Nx, int Ny, int Nz, double* in, double* out, int is_r2c, int fftw_flag, int is_oned, int is_a2a, int is_equalxy, int ah_strategy, int max_loop, int tuning_mode, int is_W0, int extrapolation_window, struct _offt_params *custom_params)
#endif
{
  /* setup1: parameter-independent settings: fftw 1-d fft */
  struct _offt_plan *po = (struct _offt_plan*)malloc(sizeof(struct _offt_plan));
  memset(po->t_init, 0, sizeof(double) * T_INIT_COUNT);
  //memset(po->t, 0, GES * sizeof(double));
  po->t_init[INIT_ALL] -= MPI_Wtime();
  po->Nx = Nx;
  po->Ny = Ny;
  po->Nz = Nz;
  MPI_Comm_size(MPI_COMM_WORLD, &(po->p));
  MPI_Comm_rank(MPI_COMM_WORLD, &(po->rank));
  po->is_r2c = is_r2c;
  po->fftw_flag = fftw_flag;
  po->is_oned = is_oned;
  po->is_a2a = is_a2a;
  po->is_equalxy = is_equalxy;
#ifdef NOTEST
  po->is_notest = is_notest;
#endif
  po->ah_strategy = ah_strategy;
  po->max_loop = max_loop;
  po->tuning_mode = tuning_mode;
  po->is_W0 = is_W0;
  po->extrapolation_window = extrapolation_window;
  po->buffers1 = NULL;
  po->buffers2 = NULL;
#ifdef P1D_LIST
  po->p1d_x_s_list = NULL;
  po->p1d_y_s_list = NULL;
  po->p1d_xy_s_list_size = 0;
  po->p1d_z = NULL;
  po->p1d_y = NULL;
  po->p1d_x = NULL;
  po->p1d_y_t = NULL;
  po->p1d_x_t = NULL;
#else // P1D_LIST
  // for fast 1-d fft
  po->t_init[INIT_FFTW] -= MPI_Wtime();
#ifdef STRIDE
  {
    int Nz_new = (po->is_r2c)? po->Nz/2+1: po->Nz;
    int rank = 1;
    int n[1] = { po->Nx };
    int howmany = 1;
    int dist = 0;
    int stride = Nz_new;
    po->p1d_x = fftw_plan_many_dft(
      rank, n, howmany,
      (fftw_complex*)out, n, (po->Ny/po->p)*stride, dist,
      (fftw_complex*)out, n, (po->Ny/po->p)*stride, dist,
      FFTW_FORWARD, po->fftw_flag);
#if 0
    po->p1d_x = fftw_plan_many_dft(
      rank, n, howmany,
      (fftw_complex*)out, n, stride, dist,
      (fftw_complex*)out, n, stride, dist,
      FFTW_FORWARD, po->fftw_flag);
#endif
    n[0] = po->Ny;
    po->p1d_y = fftw_plan_many_dft(
      rank, n, howmany,
      (fftw_complex*)out, n, stride, dist,
      (fftw_complex*)out, n, stride, dist,
      FFTW_FORWARD, po->fftw_flag);
  }
#else // STRIDE
  po->p1d_x = fftw_plan_dft_1d(Nx,
    (fftw_complex*)out, (fftw_complex*)out,
    FFTW_FORWARD, fftw_flag);
  po->p1d_y = fftw_plan_dft_1d(Ny,
    (fftw_complex*)out, (fftw_complex*)out,
    FFTW_FORWARD, fftw_flag);
#endif // STRIDE
  if (is_r2c) {
    po->p1d_z = fftw_plan_dft_r2c_1d(Nz,
      (double*)out, (fftw_complex*)out, fftw_flag);
  } else {
    po->p1d_z = fftw_plan_dft_1d(Nz,
      (fftw_complex*)out, (fftw_complex*)out,
      FFTW_FORWARD, fftw_flag);
  }
#if 0
  if (!po->rank) {
    printf("==z\n");
    fftw_print_plan(po->p1d_z);
    printf("\n");
    printf("==y\n");
    fftw_print_plan(po->p1d_y);
    printf("\n");
    printf("==x\n");
    fftw_print_plan(po->p1d_x);
    printf("\n");
  }
#endif
  po->t_init[INIT_FFTW] += MPI_Wtime();
#endif // P1D_LIST
#ifdef TRANSPOSE_LIST
  po->pt_transpose_list = NULL;
  po->pt_transpose_list_size = 0;
#endif
  po->pt_transpose = NULL;
#ifdef AVOID_TILE
  po->bad_tile_count = 0;
#endif

  /* setup2: define parameters by auto-tuning or manual setting */
  po->params = (struct _offt_params*)malloc(sizeof(struct _offt_params));
  if (po->max_loop == 0) {
    /* no auto-tuning: use default or custom parameters */
    params_set_default(po); /* Determine a default point fitting a point grid. */
    if (!po->rank) print_params(po->params->v);
    set_params_custom(po, custom_params); /* A custom point need not fit a point grid */
  } else {
    po->t_init[INIT_AH] -= MPI_Wtime();
    /* auto-tuning with active harmony */
    params_set_default(po); /* Determine a default point fitting a point grid. */
    if (!po->rank) print_params(po->params->v);
    /* set_params_custom(po, custom_params); */ /* TODO: custom values must be adjusted for tuning */
#if 0
    /* warm-up */
    if (!comm->rank) {
      printf("Warm-Up\n");
      print_params(po->params->v);
    }
    set_buffer(po);
    offt_3d_execute(po, in, out, 1);
    clear_buffer(po);
    if (!comm->rank)
      offt_print_time(po->t);
#endif
#ifdef BUFMALLOC
    set_buffer_chunk(po, 1);
#endif
#if 1
    if (ah_tuning(po, in, out) < 0) {
      printf("Error Detected during Auto-Tuning Procedure\n");
      exit(-1);
    }
#endif
#ifdef BUFMALLOC
    MPI_Free_mem(po->buffer_chunk);
    //free(po->buffer_chunk);
    po->buffer_chunk = NULL;
#endif
    po->t_init[INIT_AH] += MPI_Wtime();
  } /* end of if max_loop */

  /* setup3: parameter-dependent settings: _offt_comm, buffer, fftw transpose */
  po->comm = offt_comm_malloc(po);
#ifdef BUFMALLOC
  set_buffer_chunk(po, 0);
#endif
  set_buffer(po);
  if (po->params->v[_S_])
    po->pt_transpose = NULL;
  else
    po->pt_transpose = setup_transpose(po, out);
#ifdef P1D_LIST
  setup_p1d(po, out);
#endif
  po->t_init[INIT_ALL] += MPI_Wtime();
  //printf("%s: A\n", __FUNCTION__);
#if 1
  if (!po->rank) {
    printf("M1 %d M2 %d M3 %d M4 %d m1 %d m2 %d m3 %d m4 %d\n", po->comm->M1, po->comm->M2, po->comm->M3, po->comm->M4, po->comm->m1, po->comm->m2, po->comm->m3, po->comm->m4);
  }
#endif
  return po;
}

void offt_3d_fin(struct _offt_plan *po)
{
#ifdef P1D_LIST
  clear_p1d(po);
#else
  fftw_destroy_plan(po->p1d_x);
  fftw_destroy_plan(po->p1d_y);
  fftw_destroy_plan(po->p1d_z);
#endif
#ifdef TRANSPOSE_LIST
  clear_transpose(po);
#else
  fftw_destroy_plan(po->pt_transpose);
#endif
  offt_comm_free(po->comm);
  clear_buffer(po);
#ifdef BUFMALLOC
  MPI_Free_mem(po->buffer_chunk);
  //free(po->buffer_chunk);
  po->buffer_chunk = NULL;
#endif
  free(po->params);
  free(po);
}

void offt_3d_execute_phase1(struct _offt_plan *po, double* in, double* out, int is_tuning) {
  double *t = po->t;
  double mpi_t = 0.0;
  struct _offt_comm* comm = po->comm;
  int i;
#ifdef A2AV
  int is_a2av = po->params->v[_V_] & 2;
  int sendcounts1[comm->p2], sdispls1[comm->p2];
  int recvcounts1[comm->p2], rdispls1[comm->p2];
  int a2a1_size = 0;;
  if (is_a2av) {
    for (i = 0; i < comm->p2; i++) {
      sdispls1[i] = (i == 0)? 0: sdispls1[i-1] + sendcounts1[i-1];
      rdispls1[i] = (i == 0)? 0: rdispls1[i-1] + recvcounts1[i-1];
      sendcounts1[i] = 2 * po->params->v[_T1_] * comm->m2 * comm->F3;
      recvcounts1[i] = 2 * po->params->v[_T1_] * comm->F2 * comm->m3;
      if (i >= (comm->p2 - comm->b3))
        sendcounts1[i] += (2 * po->params->v[_T1_] * comm->m2);
      if (i >= (comm->p2 - comm->b2))
        recvcounts1[i] += (2 * po->params->v[_T1_] * comm->m3);
    }
  } else {
    a2a1_size = 2 * po->params->v[_T1_] * comm->M2 * comm->M3;
  }
#else
  int a2a1_size = 2 * po->params->v[_T1_] * comm->M2 * comm->M3;
#endif
  struct _offt_buffer** buffers = (struct _offt_buffer**)(po->buffers1);
  int myblocks = comm->m1;
  int tiling = po->params->v[_T1_], blocks = (myblocks+tiling-1)/tiling;
  int wp = po->params->v[_W1_] + 1, w = wp - 1; // w is window size, wp w+1
#ifdef FAST_TUNING
  double t_tile = 0.0;
  int tile_begin = w;
  int tile_window = po->extrapolation_window;
#endif
  for (i = 0; i < blocks; i++) {
#ifdef FAST_TUNING
    if (tile_window > 0 && is_tuning) {
      if (tile_begin <= i && i < tile_begin + tile_window) {
        t_tile -= MPI_Wtime();
      }
      if (i >= tile_begin + tile_window) {
        t[ALL] += (t_tile / tile_window);
        continue;
      }
    }
#endif
    int myT = tiling;
#ifdef A2AV
    if (i == blocks - 1) { // for the last block in case N is not divisible by T
      myT = myblocks - (blocks-1)*tiling;
      if (is_a2av) {
        int j;
        for (j = 0; j < comm->p2; j++) {
          sdispls1[j] = (j == 0)? 0: sdispls1[j-1] + sendcounts1[j-1];
          rdispls1[j] = (j == 0)? 0: rdispls1[j-1] + recvcounts1[j-1];
          sendcounts1[j] = 2 * myT * comm->m2 * comm->F3;
          recvcounts1[j] = 2 * myT * comm->F2 * comm->m3;
          if (j >= (comm->p2 - comm->b3))
            sendcounts1[j] += (2 * myT * comm->m2);
          if (j >= (comm->p2 - comm->b2))
            recvcounts1[j] += (2 * myT * comm->m3);
        }
      } else {
        a2a1_size = 2 * myT * comm->M2 * comm->M3;
      }
    }
#else
    if (i == blocks - 1) { // for the last block in case N is not divisible by T
      myT = myblocks - (blocks-1)*tiling;
      a2a1_size = 2 * myT * comm->M2 * comm->M3;
    }
#endif
    t[PACK1] -= MPI_Wtime();
    compute_fftz_pack1(out, po, i, myT);
    mpi_t = MPI_Wtime();
    t[PACK1] += mpi_t;
    if (w == 0) {
      if (po->is_a2a) {
        t[INIT1] -= mpi_t;
#ifdef A2AV
        if (is_a2av) {
          communicate_a2av(buffers[i%wp], 1, comm->comm1, sendcounts1, sdispls1, recvcounts1, rdispls1);
        } else {
          communicate_a2a(buffers[i%wp], 1, comm->comm1, a2a1_size);
        }
#else
        communicate_a2a(buffers[i%wp], 1, comm->comm1, a2a1_size);
#endif
        mpi_t = MPI_Wtime();
        t[INIT1] += mpi_t;
      } else {
        t[INIT1] -= mpi_t;
#ifdef A2AV
        if (is_a2av) {
          communicate_a2av(buffers[i%wp], 0, comm->comm1, sendcounts1, sdispls1, recvcounts1, rdispls1);
        } else {
          communicate_a2a(buffers[i%wp], 0, comm->comm1, a2a1_size);
        }
#else
        communicate_a2a(buffers[i%wp], 0, comm->comm1, a2a1_size);
#endif
        mpi_t = MPI_Wtime();
        t[INIT1] += mpi_t;
        t[WAIT1] -= mpi_t;
        communicate_wait(buffers[i%wp]);
        mpi_t = MPI_Wtime();
        t[WAIT1] += mpi_t;
      }
      t[UNPACK1] -= mpi_t;
      compute_unpack1_ffty(out, po, i, myT);
      t[UNPACK1] += MPI_Wtime();
    } else {
      if (i >= w) {
        t[WAIT1] -= mpi_t;
        communicate_wait(buffers[(i-w)%wp]);
        mpi_t = MPI_Wtime();
        t[WAIT1] += mpi_t;
        t[INIT1] -= mpi_t;
#ifdef A2AV
        if (is_a2av) {
          communicate_a2av(buffers[i%wp], 0, comm->comm1, sendcounts1, sdispls1, recvcounts1, rdispls1);
        } else {
          communicate_a2a(buffers[i%wp], 0, comm->comm1, a2a1_size);
        }
#else
        communicate_a2a(buffers[i%wp], 0, comm->comm1, a2a1_size);
#endif
        mpi_t = MPI_Wtime();
        t[INIT1] += mpi_t;
        t[UNPACK1] -= mpi_t;
        compute_unpack1_ffty(out, po, i-w, tiling);
        t[UNPACK1] += MPI_Wtime();
      } else {
        t[INIT1] -= mpi_t;
#ifdef A2AV
        if (is_a2av) {
          communicate_a2av(buffers[i%wp], 0, comm->comm1, sendcounts1, sdispls1, recvcounts1, rdispls1);
        } else {
          communicate_a2a(buffers[i%wp], 0, comm->comm1, a2a1_size);
        }
#else
        communicate_a2a(buffers[i%wp], 0, comm->comm1, a2a1_size);
#endif
        t[INIT1] += MPI_Wtime();
      }
    }
#ifdef FAST_TUNING
    if (tile_window > 0 && is_tuning) {
      if (tile_begin <= i && i < tile_begin + tile_window) {
        t_tile += MPI_Wtime();
      }
    }
#endif
  } // end of for (i) block
  int last_begin, last_end;
  last_begin = max(blocks - w, 0); last_end = blocks;
#ifdef FAST_TUNING
  if (tile_window > 0 && is_tuning) {
    if (blocks >= tile_begin + tile_window) {
      last_begin = tile_begin + tile_window - w; last_end = tile_begin + tile_window;
    }
  }
#endif
  // Waiting for the outstanding block communication
  //for (i = max(blocks - w, 0); i < blocks; i++) {
  for (i = last_begin; i < last_end; i++) {
    int myT = tiling;
    if (i == blocks - 1) // for the last block in case N is not divisible by T
      myT = myblocks - (blocks-1)*tiling;
    t[WAIT1] -= MPI_Wtime();
    communicate_wait(buffers[i%wp]);
    mpi_t = MPI_Wtime();
    t[WAIT1] += mpi_t;
    t[UNPACK1] -= mpi_t;
    compute_unpack1_ffty(out, po, i, myT);
    t[UNPACK1] += MPI_Wtime();
  }
}

void offt_3d_execute_phase2(struct _offt_plan *po, double* in, double* out, int is_tuning) {
  double *t = po->t;
  double mpi_t = 0.0;
  struct _offt_comm* comm = po->comm;
  int i;
#ifdef A2AV
  int is_a2av = po->params->v[_V_] & 1;
  int sendcounts2[comm->p1], sdispls2[comm->p1];
  int recvcounts2[comm->p1], rdispls2[comm->p1];
  int a2a2_size = 0;
  if (is_a2av) {
    for (i = 0; i < comm->p1; i++) {
      sdispls2[i] = (i == 0)? 0: sdispls2[i-1] + sendcounts2[i-1];
      rdispls2[i] = (i == 0)? 0: rdispls2[i-1] + recvcounts2[i-1];
      sendcounts2[i] = 2 * po->params->v[_T2_] * comm->m1 * comm->F4;
      recvcounts2[i] = 2 * po->params->v[_T2_] * comm->F1 * comm->m4;
      if (i >= (comm->p1 - comm->b4))
        sendcounts2[i] += (2 * po->params->v[_T2_] * comm->m1);
      if (i >= (comm->p1 - comm->b1))
        recvcounts2[i] += (2 * po->params->v[_T2_] * comm->m4);
    }
  } else {
    a2a2_size = 2 * po->params->v[_T2_] * comm->M1 * comm->M4;
  }
#else
  int a2a2_size = 2 * po->params->v[_T2_] * comm->M1 * comm->M4;
#endif
  struct _offt_buffer** buffers = (struct _offt_buffer**)(po->buffers2);
  int myblocks = comm->m3;
  int tiling = po->params->v[_T2_], blocks = (myblocks+tiling-1)/tiling;
  int wp = po->params->v[_W2_] + 1, w = wp - 1; // w is window size, wp w+1
#ifdef FAST_TUNING
  double t_tile = 0.0;
  int tile_begin = w;
  int tile_window = po->extrapolation_window;
#endif
  for (i = 0; i < blocks; i++) {
#ifdef FAST_TUNING
    //if (i != blocks - 1 && tile_window > 0 && is_tuning) {
    if (tile_window > 0 && is_tuning) {
      if (tile_begin <= i && i < tile_begin + tile_window) {
        t_tile -= MPI_Wtime();
      }
      if (i >= tile_begin + tile_window) {
        t[ALL] += (t_tile / tile_window);
        continue;
      }
    }
#endif
    int myT = tiling;
#ifdef A2AV
    if (i == blocks - 1) { // for the last block in case N is not divisible by T
      myT = myblocks - (blocks-1)*tiling;
      if (is_a2av) {
        int j;
        for (j = 0; j < comm->p1; j++) {
          sdispls2[j] = (j == 0)? 0: sdispls2[j-1] + sendcounts2[j-1];
          rdispls2[j] = (j == 0)? 0: rdispls2[j-1] + recvcounts2[j-1];
          sendcounts2[j] = 2 * myT * comm->m1 * comm->F4;
          recvcounts2[j] = 2 * myT * comm->F1 * comm->m4;
          if (j >= (comm->p1 - comm->b4))
            sendcounts2[j] += (2 * myT * comm->m1);
          if (j >= (comm->p1 - comm->b1))
            recvcounts2[j] += (2 * myT * comm->m4);
        }
      } else {
        a2a2_size = 2 * myT * comm->M1 * comm->M4;
      }
    }
#else
    if (i == blocks - 1) { // for the last block in case N is not divisible by T
      myT = myblocks - (blocks-1)*tiling;
      a2a2_size = 2 * myT * comm->M1 * comm->M4;
    }
#endif
    t[PACK2] -= MPI_Wtime();
    compute_ffty_pack2(out, po, i, myT);
    mpi_t = MPI_Wtime();
    t[PACK2] += mpi_t;
    if (w == 0) {
      if (po->is_a2a) {
        t[INIT2] -= mpi_t;
#ifdef A2AV
        if (is_a2av) {
          communicate_a2av(buffers[i%wp], 1, comm->comm2, sendcounts2, sdispls2, recvcounts2, rdispls2);
        } else {
          communicate_a2a(buffers[i%wp], 1, comm->comm2, a2a2_size);
        }
#else
        communicate_a2a(buffers[i%wp], 1, comm->comm2, a2a2_size);
#endif
        mpi_t = MPI_Wtime();
        t[INIT2] += mpi_t;
      } else {
        t[INIT2] -= mpi_t;
#ifdef A2AV
        if (is_a2av) {
          communicate_a2av(buffers[i%wp], 0, comm->comm2, sendcounts2, sdispls2, recvcounts2, rdispls2);
        } else {
          communicate_a2a(buffers[i%wp], 0, comm->comm2, a2a2_size);
        }
#else
        communicate_a2a(buffers[i%wp], 0, comm->comm2, a2a2_size);
#endif
        mpi_t = MPI_Wtime();
        t[INIT2] += mpi_t;
        t[WAIT2] -= mpi_t;
        communicate_wait(buffers[i%wp]);
        mpi_t = MPI_Wtime();
        t[WAIT2] += mpi_t;
      }
      t[UNPACK2] -= mpi_t;
      compute_unpack2_fftx(out, po, i, myT);
      t[UNPACK2] += MPI_Wtime();
    } else {
      if (i >= w) {
        t[WAIT2] -= mpi_t;
        communicate_wait(buffers[(i-w)%wp]);
        mpi_t = MPI_Wtime();
        t[WAIT2] += mpi_t;
        t[INIT2] -= mpi_t;
#ifdef A2AV
        if (is_a2av) {
          communicate_a2av(buffers[i%wp], 0, comm->comm2, sendcounts2, sdispls2, recvcounts2, rdispls2);
        } else {
          communicate_a2a(buffers[i%wp], 0, comm->comm2, a2a2_size);
        }
#else
        communicate_a2a(buffers[i%wp], 0, comm->comm2, a2a2_size);
#endif
        mpi_t = MPI_Wtime();
        t[INIT2] += mpi_t;
        t[UNPACK2] -= mpi_t;
        compute_unpack2_fftx(out, po, i-w, tiling);
        t[UNPACK2] += MPI_Wtime();
      } else {
        t[INIT2] -= mpi_t;
#ifdef A2AV
        if (is_a2av) {
          communicate_a2av(buffers[i%wp], 0, comm->comm2, sendcounts2, sdispls2, recvcounts2, rdispls2);
        } else {
          communicate_a2a(buffers[i%wp], 0, comm->comm2, a2a2_size);
        }
#else
        communicate_a2a(buffers[i%wp], 0, comm->comm2, a2a2_size);
#endif
        t[INIT2] += MPI_Wtime();
      }
    }
#ifdef FAST_TUNING
    if (tile_window > 0 && is_tuning) {
      if (tile_begin <= i && i < tile_begin + tile_window) {
        t_tile += MPI_Wtime();
      }
    }
#endif
  } // end of for (i) block
  int last_begin, last_end;
  last_begin = max(blocks - w, 0); last_end = blocks;
#ifdef FAST_TUNING
  if (tile_window > 0 && is_tuning) {
    if (blocks >= tile_begin + tile_window) {
      last_begin = tile_begin + tile_window - w; last_end = tile_begin + tile_window;
    }
  }
#endif
  // Waiting for the outstanding block communication
  //for (i = max(blocks - w, 0); i < blocks; i++) {
  for (i = last_begin; i < last_end; i++) {
    int myT = tiling;
    if (i == blocks - 1) // for the last block in case N is not divisible by T
      myT = myblocks - (blocks-1)*tiling;
    t[WAIT2] -= MPI_Wtime();
    communicate_wait(buffers[i%wp]);
    mpi_t = MPI_Wtime();
    t[WAIT2] += mpi_t;
    t[UNPACK2] -= mpi_t;
    compute_unpack2_fftx(out, po, i, myT);
    t[UNPACK2] += MPI_Wtime();
  }
}

void offt_3d_execute(struct _offt_plan *po, double* in, double* out, int is_tuning)
{
  // TODO: out-of-place, in must be equal to out now.
  double *t = po->t;
  memset(t, 0, GES * sizeof(double));
#ifdef AVOID_TILE
  if (!is_tuning) {
    int p1 = po->params->v[_P1_];
    int p2 = po->p / p1;
    struct _offt_comm *comm = (struct _offt_comm*)po->comm;
    long T1_sz = po->params->v[_T1_] * comm->M2 * ((po->is_r2c)?po->Nz/2+1:po->Nz);
    long T2_sz = comm->M1 * po->Ny * po->params->v[_T2_];
    if ((ERROR_LOW <= T1_sz/p2 && T1_sz/p2 <= ERROR_HIGH) ||
        (ERROR_LOW <= T2_sz/p1 && T2_sz/p1 <= ERROR_HIGH)) {
      if (po->bad_tile_count >= MAX_BAD_TILE_COUNT + 1) {
        if (!po->rank)
          printf("@ bad T for MPI_Alloc_mem %d skip fft\n", po->bad_tile_count);
        t[ALL] = 99999999.0;
        return;
      } else {
        po->bad_tile_count++;
        if (!po->rank)
          printf("@ bad T for MPI_Alloc_mem %d ignore\n", po->bad_tile_count);
      }
    }
  }
#endif
  t[ALL] -= MPI_Wtime();
  /* input layout: x-y-z */
  /* x-stride: max((M2*p2)*M3, (M4*p1)*M3) */
  /* y-stride: M3*p2 */
  /* z-stride: 1 */
  if (po->is_oned && po->comm->p1 == 1) {
    /* METHOD ONE */
    /* 1 x p */
    /* ignore parameters of *2, Fx */
    /*
     *********************************************************
     for each tile i along x
       FFTz and Pack (i) // xyz->xzyB, sub-tile on x-y plane
       wait (i-W1)
       ia2a (i)
       Unpack and FFTy (i-W1) // xzyB->xzy, sub-tile on x-z plane
     Tpose xzy->zyx or xzy->yzx
     FFTx
     *********************************************************
     */
    offt_3d_execute_phase1(po, in, out, is_tuning);
    if (!(is_tuning && po->tuning_mode == 1)) { /* skip this */
#ifdef STRIDE
      if (po->params->v[_S_]) {
      } else {
        /* xzy -> zyx */
        /* xzy -> yzx when Nx == Ny */
        t[TRANSPOSE] -= MPI_Wtime();
        fftw_execute_dft(po->pt_transpose, (fftw_complex*)out, (fftw_complex*)out);
        t[TRANSPOSE] += MPI_Wtime();
      }
#else
      /* xzy -> zyx */
      /* xzy -> yzx when Nx == Ny */
      t[TRANSPOSE] -= MPI_Wtime();
      fftw_execute_dft(po->pt_transpose, (fftw_complex*)out, (fftw_complex*)out);
      t[TRANSPOSE] += MPI_Wtime();
#endif
      t[FFTx] -= MPI_Wtime();
#ifdef STRIDE
      if (po->params->v[_S_]) {
        fftw_execute_dft(po->p1d_x, (fftw_complex*)out, (fftw_complex*)out);
      } else {
        int y, z;
        for (z = 0; z < po->comm->m3; z++)
          for (y = 0; y < po->Ny; y++) {
            double *ptr = out + 2*(po->comm->M1*po->comm->p1)*(y + po->comm->M4 * z);
            fftw_execute_dft(po->p1d_x, (fftw_complex*)ptr, (fftw_complex*)ptr);
          }
      }
#else // STRIDE
      int y, z;
      for (z = 0; z < po->comm->m3; z++)
        for (y = 0; y < po->Ny; y++) {
          double *ptr = out + 2*(po->comm->M1*po->comm->p1)*(y + po->comm->M4 * z);
          fftw_execute_dft(po->p1d_x, (fftw_complex*)ptr, (fftw_complex*)ptr);
        }
#endif // STRIDE
      t[FFTx] += MPI_Wtime();
    }
  } else if (po->is_oned && po->comm->p1 == po->p) {
    /* METHOD OLD */
    /* p x 1 */
    /* ignore parameters of *1, Fz, Py2 */
    /*
     *********************************************************
     FFTz
     Tpose xyz->zxy
     for each tile i along z
       FFTy and Pack (i) // zxy->zyxB or xzy->yzxB
       wait (i-W2)
       ia2a (i)
       Unpack and FFTx (i-W2) // zyxB->zyx or yzxB->yzx
     *********************************************************
     */
    if (!(is_tuning && po->tuning_mode == 2)) { /* skip this */
      /* FFTz for xyz */
      int x, y;
      t[FFTz] -= MPI_Wtime();
      for (x = 0; x < po->comm->m1; x++) {
        for (y = 0; y < po->Ny; y++) {
          double *ptr = out + 2*po->comm->istride[1]*y + 2*po->comm->istride[0]*x;
          if (po->is_r2c) {
            fftw_execute_dft_r2c(po->p1d_z, (double*)ptr, (fftw_complex*)ptr);
          } else {
            fftw_execute_dft(po->p1d_z, (fftw_complex*)ptr, (fftw_complex*)ptr);
          }
        }
      }
      t[FFTz] += MPI_Wtime();
#ifdef STRIDE
      if (po->params->v[_S_]) {
      } else {
        /* xyz -> zxy */
        /* xzy -> yzx when Nx == Ny */
        t[TRANSPOSE] -= MPI_Wtime();
        fftw_execute_dft(po->pt_transpose, (fftw_complex*)out, (fftw_complex*)out);
        t[TRANSPOSE] += MPI_Wtime();
      }
#else
      /* xyz -> zxy */
      /* xzy -> yzx when Nx == Ny */
      t[TRANSPOSE] -= MPI_Wtime();
      fftw_execute_dft(po->pt_transpose, (fftw_complex*)out, (fftw_complex*)out);
      t[TRANSPOSE] += MPI_Wtime();
#endif
    } /* end of if !tuning_mode == 2 */
    offt_3d_execute_phase2(po, in, out, is_tuning);
  } else {
    /* p1 x p2 */
    /*
     *********************************************************
     for each tile i along x
       FFTz and Pack (i) // xyz->xzyB, sub-tile on x-y plane
       wait (i-W1)
       ia2a (i)
       Unpack and FFTy (i-W1) // xzyB->xzy, sub-tile on x-z plane
     for each tile j along z
       Pack (j) // zxy->zyxB or xzy->yzxB
       wait (j-W2)
       ia2a (j)
       Unpack and FFTx (j-W2) // zyxB->zyx or yzxB->yzx
     *********************************************************
     */
    offt_3d_execute_phase1(po, in, out, is_tuning);
#ifdef STRIDE
    if (po->params->v[_S_]) {
    } else {
      if (!(po->is_equalxy && po->comm->M1 == po->comm->M4)) {
        /* xzy => zxy for tiling, in-place, Nx != Ny */
        /* xzy, no transpose when Nx == Ny */
        t[TRANSPOSE] -= MPI_Wtime();
        fftw_execute_dft(po->pt_transpose, (fftw_complex*)out, (fftw_complex*)out);
        t[TRANSPOSE] += MPI_Wtime();
      }
    }
#else
    if (!(po->is_equalxy && po->comm->M1 == po->comm->M4)) {
      /* xzy => zxy for tiling, in-place, Nx != Ny */
      /* xzy, no transpose when Nx == Ny */
      t[TRANSPOSE] -= MPI_Wtime();
      fftw_execute_dft(po->pt_transpose, (fftw_complex*)out, (fftw_complex*)out);
      t[TRANSPOSE] += MPI_Wtime();
    }
#endif
    offt_3d_execute_phase2(po, in, out, is_tuning);
  } /* end of if is_oned */
  /* output layout: z-y-x */
  /* z-stride: (M1*p1)*M4 */
  /* y-stride: M1*p1 */
  /* x-stride: 1 */
  /* output layout: y-z-x when Nx == Ny */
  /* y-stride: (M1*p1)*M3 */
  /* z-stride: M1*p1 */
  /* x-stride: 1 */
  t[ALL] += MPI_Wtime();
  return;
}
