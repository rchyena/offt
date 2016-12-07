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

#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "offt-internal.h"

#ifdef AH_TUNING
#include "hclient.h"
#include "hmesg.h"
#include "hsession.h"
#include "hval.h"
#include <fcntl.h>
#include <signal.h>
#include <unistd.h>

pid_t launch_silent(const char *prog, char **argv)
{
    int i, fd;
    pid_t pid;

    fd = open("/dev/null", O_WRONLY);
    if (fd < 0) {
        perror("Error opening /dev/null");
        return -1;
    }

    {
        printf("Launching %s", prog);
        for (i = 1; argv[i] != NULL; ++i) {
            printf(" %s", argv[i]);
        }
        printf(" > /dev/null\n");
    }

    pid = fork();
    if (pid == 0) {
        /* Child Case */
        if (dup2(fd, STDOUT_FILENO) < 0 ||
            dup2(fd, STDERR_FILENO) < 0)
        {
            perror("Could not redirect stdout or stderr via dup2()");
            exit(-1);
        }
        close(fd);

        execv(prog, argv);
        exit(-2);
    }
    else if (pid < 0)
        perror("Error on fork()");

    close(fd);
    return pid;
}

/* **********************************************************
 @ parameter conversion
   ********************************************************** */
void params_convert(int is_backward, int* v, long* ahv, struct _offt_plan* po, int** v_list, int* v_list_size) {
  int i = 0;
  if (is_backward) {
    for (i = 0; i < PARAM_COUNT; i++) {
      if (ahv[i] > v_list_size[i]) {
        printf("params_convert: bwd OUT OF RANGE ERROR\n");
        exit(-1);
      }
      v[i] = v_list[i][ahv[i]];
    }
#ifdef ADJUST_POINT
    if (po->is_oned && v[_P1_] == 1) {
      v[_Ry_] = 10;
      v[_T2_] = 1; v[_W2_] = 0;
      v[_Fy2_] = v[_FP2_] = v[_FU2_] = v[_Fx_] = 0;
      v[_Pz2_] = v[_Px2_] = v[_Uz2_] = v[_Uy2_] = 1;
    }
    if (po->is_oned && v[_P1_] == po->p) {
      v[_Ry_] = 0;
      v[_T1_] = 1; v[_W1_] = 0;
      v[_Fz_] = v[_FP1_] = v[_FU1_] = v[_Fy1_] = 0;
      v[_Px1_] = v[_Py1_] = v[_Ux1_] = v[_Uz1_] = 1;
    }
    if (v[_W1_] == 0) {
      v[_Fz_] = v[_FP1_] = v[_Fy1_] = v[_FU1_] = 0;
    }
    if (v[_W2_] == 0) {
      v[_Fy2_] = v[_FP2_] = v[_Fx_] = v[_FU2_] = 0;
    }
    {
      int Nz_new = (po->is_r2c)? po->Nz/2 + 1: po->Nz;
      int p1 = v[_P1_];
      int p2 = po->p / p1;
      /* remove left bit to use alltoall in phase I */
      if (po->Ny % p2 == 0 && Nz_new % p2 == 0) v[_V_] = v[_V_] & 1;
      /* remove right bit to use alltoall in phase II */
      if (po->Nx % p1 == 0 && po-> Ny % p1 == 0) v[_V_] = v[_V_] & 2;
    }
#endif
  } else {
    for (i = 0; i < PARAM_COUNT; i++) {
      int c;
      int found_c = -1;
      for (c = 0; c < v_list_size[i]; c++) {
        if (v_list[i][c] == v[i]) {
          found_c = c;
          break;
        }
      }
      if (found_c == -1) {
        printf("params_convert: fwd OUT OF RANGE ERROR\n");
        exit(-1);
      }
      ahv[i] = found_c;
    }
  }
}

/* **********************************************************
 @ infeasible point check
   This function only tests if the relation among parameters is correct.
   We rely on the AH server for checking if a point is in a rectangle area 
   as the AH server knows the lower and upper bound of each parameter.
   ********************************************************** */
int is_infeasible_point(struct _offt_plan *po, int *v, int *p_i) {
  *p_i = -1;
#if 0
  if (v[_W1_] == 0 &&
     (v[_Fz_] != 0 || v[_FP1_] != 0 || v[_Fy1_] != 0 || v[_FU1_] != 0)) {
    *p_i = _W1_;
    return 1;
  }
  if (v[_W2_] == 0 &&
     (v[_Fy2_] != 0 || v[_FP2_] != 0 || v[_Fx_] != 0 || v[_FU2_] != 0)) {
    *p_i = _W2_;
    return 1;
  }
#endif
  int p = po->p;
  int p1 = v[_P1_];
  int p2 = p / p1;
  int Nz_new = (po->is_r2c)?po->Nz/2+1:po->Nz;
  int M1 = (po->Nx+p1-1)/p1;
  int M2 = (po->Ny+p2-1)/p2;
  int M3 = (Nz_new+p2-1)/p2;
  int M4 = (po->Ny+p1-1)/p1;
  if (p1 > po->Nx || p1 > po->Ny || p2 > po->Ny || p2 > Nz_new)  { *p_i = _P1_; return 1; }
  if (v[_T1_] < 1 || M1 < v[_T1_]) { *p_i = _T1_; return 1; }
  if ((M1+v[_T1_]-1)/v[_T1_] < v[_W1_] ||
      (M1 == v[_T1_] && v[_W1_] > 0) ||
      (v[_T1_] * M2 * (M3 * p2) > BUFFER_SIZE_LIMIT / (v[_W1_]+1) / 2 / 2)) /* 2: complex 2: a2as+a2ar */
  { *p_i = _W1_; return 1; }
  if (v[_Px1_] < 1 || v[_T1_] < v[_Px1_]) { *p_i = _Px1_; return 1; }
  if (v[_Py1_] < 1 || M2 < v[_Py1_]) { *p_i = _Py1_; return 1; }
  if (v[_Fz_] < 0 || v[_T1_]*M2 < v[_Fz_]) { *p_i = _Fz_; return 1; }
  if (v[_FP1_] < 0 || v[_T1_]/v[_Px1_]*M2/v[_Py1_] < v[_FP1_]) { *p_i = _FP1_; return 1; }
  if (v[_Ux1_] < 1 || v[_T1_] < v[_Ux1_]) { *p_i = _Ux1_; return 1; }
  if (v[_Uz1_] < 1 || M3 < v[_Uz1_]) { *p_i = _Uz1_; return 1; }
  if (v[_Fy1_] < 0 || v[_T1_]*M3 < v[_Fy1_]) { *p_i = _Fy1_; return 1; }
  if (v[_FU1_] < 0 || v[_T1_]/v[_Ux1_]*M3/v[_Uz1_] < v[_FU1_]) { *p_i = _FU1_; return 1; }
  if (v[_T2_] < 1 || M3 < v[_T2_]) { *p_i = _T2_; return 1; }
  if ((M3+v[_T2_]-1)/v[_T2_] < v[_W2_] ||
      (M3 == v[_T2_] && v[_W2_] > 0) ||
      (M1 * (M4 * p1) * v[_T2_] > BUFFER_SIZE_LIMIT / (v[_W2_]+1) / 2 / 2)) /* 2: complex 2: a2as+a2ar */
  { *p_i = _W2_; return 1; }
  if (v[_Fy2_] < 0 || v[_T2_]*M1 < v[_Fy2_]) { *p_i = _Fy2_; return 1; }
  if (v[_Px2_] < 1 || M1 < v[_Px2_]) { *p_i = _Px2_; return 1; }
#if 0
  if (v[_Ry_] > min(10, M1) ||
      (po->is_oned && v[_P1_] == 1 && v[_Ry_] < min(10, M1)) ||
      (po->is_oned && v[_P1_] == po->p && v[_Ry_] != 0)) { *p_i = _Ry_; return 1; }
#endif
  if (v[_Pz2_] < 1 || v[_T2_] < v[_Pz2_]) { *p_i = _Pz2_; return 1; }
#if 1 /* in case F is too small */
  if (v[_FP2_] < 0) { *p_i = _FP2_; return 1; }
#else
  if (v[_FP2_] < 0 || M1/v[_Px2_]*v[_T2_]/v[_Pz2_] < v[_FP2_]) { *p_i = _FP2_; return 1; }
#endif
  //if (v[_FP2_] < 0 || M1/v[_Px2_]*M2/v[_Py2_]*v[_T2_]/v[_Pz2_] < v[_FP2_]) { *p_i = _FP2_; return 1; }
  if (v[_Uy2_] < 1 || M4 < v[_Uy2_]) { *p_i = _Uy2_; return 1; }
  if (v[_Uz2_] < 1 || v[_T2_] < v[_Uz2_]) { *p_i = _Uz2_; return 1; }
#if 1 /* in case F is too small */
  if (v[_FP2_] < 0) { *p_i = _FP2_; return 1; }
#else
  if (v[_FU2_] < 0 || M4/v[_Uy2_]*v[_T2_]/v[_Uz2_] < v[_FU2_]) { *p_i = _FU2_; return 1; }
#endif
  if (v[_Fx_] < 0 || v[_T2_]*M4 < v[_Fx_]) { *p_i = _Fx_; return 1; }

  if (v[_V_] < 0 || v[_V_] > 3) { *p_i = _V_; return 1; }
  if (v[_S_] < 0 || v[_S_] > 1) { *p_i = _S_; return 1; }
#ifdef AVOID_TILE
  long T1_sz = po->params->v[_T1_] * M2 * ((po->is_r2c)?po->Nz/2+1:po->Nz);
  long T2_sz = M1 * po->Ny * po->params->v[_T2_];
  if ((ERROR_LOW <= T1_sz/p2 && T1_sz/p2 <= ERROR_HIGH) ||
      (ERROR_LOW <= T2_sz/p1 && T2_sz/p1 <= ERROR_HIGH)) {
    if (po->bad_tile_count >= MAX_BAD_TILE_COUNT) {
      printf("@ bad T for MPI_Alloc_mem %d return infeasible\n", po->bad_tile_count);
      return 1;
    } else {
      po->bad_tile_count++;
      printf("@ bad T for MPI_Alloc_mem %d ignore\n", po->bad_tile_count);
    }
  }
#endif

  return 0;
}

/* **********************************************************
 @ point history
   ********************************************************** */
int is_in_database_point(struct _offt_plan *po, double *pperf) {
  //int x[PARAM_COUNT]; // current point
  //convert_params(0, (struct _offt_comm*)po->comm, po->params->v, x);
  int* x = po->params->v;
  FILE *f;
  f = fopen(po->point_database_file, "r");
  int is_found = 1;
  if (f == NULL) {
    is_found = 0;
  } else {
    while (!feof(f)) {
      double read_perf;
      int y[PARAM_COUNT];
      int j;
      fscanf(f, "%lf", &read_perf);
      is_found = 1;
      for (j = 0; j < PARAM_COUNT; j++) {
        fscanf(f, "%d", &y[j]);
        //if (!po->rank) printf("%.3f %d %d\n", read_perf, x[j], y[j]);
        if (x[j] != y[j])
          is_found = 0;
      }
      if (is_found) {
        *pperf = read_perf;
        break;
      }
    }
    fclose(f);
  }
  
  /*if (!po->rank) printf("%s: is_found %d\n", __FUNCTION__, is_found);*/
  return is_found;
}

void write_to_database(struct _offt_plan *po, double perf) {
  //int x[PARAM_COUNT];
  int *x = po->params->v;
  FILE *f;
  int j;
  f = fopen(po->point_database_file, "a");
  fprintf(f, "%.5f ", perf);
  for (j = 0; j < PARAM_COUNT; j++) {
    fprintf(f, "%d ", x[j]);
  }
  fprintf(f, "\n");
  fclose(f);
}

/* **********************************************************
 @ write an initial simplex info to a file
   ********************************************************** */
#if 0
/* deterministic initial simplex */
void write_initial_simplex(struct _offt_plan *po, int** v_list, int *v_list_size)
{
  long ahv[PARAM_COUNT];
  /* po->params contain the default parameter values. */
  params_convert(0, po->params->v, ahv, po, v_list, v_list_size);
  int x[PARAM_COUNT+1][PARAM_COUNT];
  for (int i = 0; i < PARAM_COUNT + 1; i++)
    for (int j = 0; j < PARAM_COUNT; j++)
      x[i][j] = ahv[j];

  /* define other simplex points around the default point. */
  int k = PARAM_COUNT / 2;
  int diff = 2;
  int i;
  for (i = 1; i < PARAM_COUNT + 1; i++) {
    int jj;
    for (jj = i - 1; jj <= i - 1 + k - 1; jj++) {
      int j = (jj + PARAM_COUNT) % PARAM_COUNT;
      x[i][j] = (i % 2) ? x[0][j] + diff
                        : x[0][j] - diff;
      /* adjust it in a hyperrectangle */
      x[i][j] = min(max(x[i][j], 0), v_list_size[j]-1);
    }
  }

  /* make all initial points feasible */
  /* assume the default base point is feasible */
  int p = po->p;
  int Nz_new = (po->is_r2c)?po->Nz/2+1:po->Nz;
  for (i = 1; i < PARAM_COUNT + 1; i++) {
    int j; /* j: parameter index */
    int vv[PARAM_COUNT]; /* real point of ah point x[i] */
    for (j = 0; j < PARAM_COUNT; j++) ahv[j] = (long)x[i][j];
    params_convert(1, vv, ahv, po, v_list, v_list_size);
    for (j = 0; j < PARAM_COUNT; j++) {
      /* make each parameter feasible */
      /* TODO: three redundant definitions of lower and upper bound: default, is_infeasible, here */
      int p2 = p / vv[_P1_];
      /* TODO: redundant definitions of M1-M4 */
      int M1 = (po->Nx + vv[_P1_] - 1) / vv[_P1_];
      int M2 = (po->Ny + p2 - 1) / p2;
      int M3 = (Nz_new + p2 - 1) / p2;
      int M4 = (po->Ny + vv[_P1_] - 1) / vv[_P1_];
      switch (j) {
      case _P1_: break;
      case _T1_: vv[j] = min(max(vv[j],1),M1); break;
      case _W1_: vv[j] = min(max(vv[j],0),(M1+vv[_T1_]-1)/vv[_T1_]); break;
      case _Px1_: vv[j] = min(max(vv[j],1),vv[_T1_]); break;
      case _Py1_: vv[j] = min(max(vv[j],1),M2); break;
      case _Fz_: vv[j] = min(max(vv[j],0),vv[_T1_]*M2); break;
      case _FP1_: vv[j] = min(max(vv[j],0),vv[_T1_]/vv[_Px1_]*M2/vv[_Py1_]); break;
      case _Ux1_: vv[j] = min(max(vv[j],1),vv[_T1_]); break;
      case _Uz1_: vv[j] = min(max(vv[j],1),M3); break;
      case _FU1_: vv[j] = min(max(vv[j],0),vv[_T1_]/vv[_Ux1_]*M3/vv[_Uz1_]); break;
      case _Fy1_: vv[j] = min(max(vv[j],0),vv[_T1_]*M3); break;
      case _T2_: vv[j] = min(max(vv[j],1),M3); break;
      case _W2_: vv[j] = min(max(vv[j],0),(M3+vv[_T2_]-1)/vv[_T2_]); break;
      case _Fy2_: vv[j] = min(max(vv[j],0),vv[_T2_]*M1); break;
      case _Px2_: vv[j] = min(max(vv[j],1),M1); break;
      case _Ry_: break;
      case _Pz2_: vv[j] = min(max(vv[j],1),vv[_T2_]); break;
      case _FP2_: vv[j] = min(max(vv[j],0),M1/vv[_Px2_]*vv[_T2_]/vv[_Pz2_]); break;
      //case _FP2_: vv[j] = min(max(vv[j],0),M1/vv[_Px2_]*M2/vv[_Py2_]*vv[_T2_]/vv[_Pz2_]); break;
      case _Uy2_: vv[j] = min(max(vv[j],1),M4); break;
      case _Uz2_: vv[j] = min(max(vv[j],1),vv[_T2_]); break;
      case _FU2_: vv[j] = min(max(vv[j],0),M4/vv[_Uy2_]*vv[_T2_]/vv[_Uz2_]); break;
      case _Fx_: vv[j] = min(max(vv[j],0),vv[_T2_]*M4); break;
      }
      switch (j) {
      case _Fz_:
      case _FP1_:
      case _FU1_:
      case _Fy1_:
        if (vv[_W1_] == 0) vv[j] = 0;
        break;
      case _Fy2_:
      case _FP2_:
      case _FU2_:
      case _Fx_:
        if (vv[_W2_] == 0) vv[j] = 0;
        break;
      }
      /* fit in the grid */
      vv[j] = grid_value_floor(0, v_list, v_list_size, j, vv[j]);
    }
    params_convert(0, vv, ahv, po, v_list, v_list_size);
    for (j = 0; j < PARAM_COUNT; j++) x[i][j] = (int)ahv[j];
  }

  FILE *f;
  f = fopen(po->user_vertex_file, "w");
  for (int i = 0; i < PARAM_COUNT + 1; i++) {
    for (int j = 0; j < PARAM_COUNT; j++) {
      fprintf(f, "%d ", x[i][j]);
    }
    fprintf(f, "\n");
  }
  fclose(f);
}
#else /* deterministic */
#if 0 /* pure random */
/* randomized initial simplex */
/* TODO: return random point */
void random_ahv_point(long *ahv, struct _offt_plan *po, int** v_list, int *v_list_size) {
  int v[PARAM_COUNT];
  int is_infeasible;
  do {
    int j;
    int err_i;
    for (j = 0; j < PARAM_COUNT; j++) {
      /* random k: v_list[j][0<=k<v_list_size[j]] */
      ahv[j] = (int)(rand() % v_list_size[j]);
    }
    /* convert bwd */
    params_convert(1, v, ahv, po, v_list, v_list_size);
    is_infeasible = is_infeasible_point(po, v, &err_i);
  } while (is_infeasible);
}

void write_initial_simplex(struct _offt_plan *po, int** v_list, int *v_list_size)
{
  long ahv[PARAM_COUNT];
  int x[PARAM_COUNT+1][PARAM_COUNT];
  int i;
  for (i = 0; i < PARAM_COUNT + 1; i++) {
    random_ahv_point(ahv, po, v_list, v_list_size); /* ahv: one random feasible point */
    for (int j = 0; j < PARAM_COUNT; j++)
      x[i][j] = ahv[j];
  }

  FILE *f;
  f = fopen(po->user_vertex_file, "w");
  for (int i = 0; i < PARAM_COUNT + 1; i++) {
    for (int j = 0; j < PARAM_COUNT; j++) {
      fprintf(f, "%d ", x[i][j]);
    }
    fprintf(f, "\n");
  }
  fclose(f);
}
#else /* pure random */
/* hybrid random */
void write_initial_simplex(struct _offt_plan *po, int** v_list, int *v_list_size)
{
  int x[PARAM_COUNT+1][PARAM_COUNT];
  int i;

  int Nz_new = (po->is_r2c)?po->Nz/2+1:po->Nz;
  for (i = 0; i < PARAM_COUNT + 1; i++) {
    int vv[PARAM_COUNT];
    int j, v_min = 0, v_max = 0, v_low, v_high, g_min, g_max;
    int p2 = 0, M1 = 0, M2 = 0, M3 = 0, M4 = 0;
    for (j = 0; j < PARAM_COUNT; j++) {
      switch (j) {
      case _P1_: /* 1-p */
        v_min = 1; v_max = po->p;
        if (po->tuning_mode == 1) {
          v_min = 1; v_max = 1;
        } else if (po->tuning_mode == 2) {
          v_min = po->p; v_max = po->p;
        }
        break;
      case _T1_: /* 1-M1, small msg overhead >=1KB (64elem) per node, buffer size overhead <=64MB (4M elems) */
        /*v_min = M1/64; v_max = M1/4;*/ /* overlap: M1/16 */
        v_min = 1; v_max = M1;
#if 1
        v_low = min(v_max, 64 * p2 / M2 / (M3 * p2)); /* small msg overhead >= 64 elems per node */
        v_min = max(v_min, v_low);
        v_high = max(v_min, (BUFFER_SIZE_LIMIT/8) / M2 / (M3 * p2)); /* buffer size overhead <= 4M elems */
        v_max = min(v_max, v_high);
#endif
        break;
      case _W1_: /* 0-10, 0-(M1+T1-1)/T1 */
        if (po->is_W0) {
          v_min = v_max = 0;
        } else {
          v_min = 0; v_max = 5;
          if (M1 == vv[_T1_]) {
            v_min = 0; v_max = 0;
          } else {
            /* 2 * 2 * (vv[_T1_] * M2 * Nz_new) * (vv[_W1_]+1) <= BUFFER_SIZE_LIMIT */
            v_high = min((M1+vv[_T1_]-1)/vv[_T1_], BUFFER_SIZE_LIMIT / 2 / 2 / (vv[_T1_] * M2 * (M3 * p2)) - 1);
            v_high = max(v_min, v_high);
            v_max = min(v_max, v_high);
          }
        }
        break;
      case _Px1_: /* 1-T1, 128-256(8192)-512KB cache, sqrt(8192/8/Nz_new - 8192*2/Nz_new) */
        v_min = 1; v_max = vv[_T1_];
        /* assume 32KB - 4MB cache */
        v_low = min(v_max, (int)sqrt(SUBTILE_SIZE/8/Nz_new));
        v_low = min(v_low, SUBTILE_SIZE/8/Nz_new/M2); /* in case M2 is large */
        v_min = max(v_min, v_low);
        v_high = max(v_min, (int)sqrt(SUBTILE_SIZE*16/Nz_new));
        v_high = max(v_high, SUBTILE_SIZE*16/Nz_new/M2); /* in case M2 is small */
        v_max = min(v_max, v_high);
        break;
      case _Py1_: /* Py1: 1-M2, sqrt(8192/2/Nz_new) */
        v_min = 1; v_max = M2;
        v_low = min(v_max, SUBTILE_SIZE/8/Nz_new/vv[_Px1_]);
        v_min = max(v_min, v_low);
        v_high = max(v_min, SUBTILE_SIZE*16/Nz_new/vv[_Px1_]);
        v_max = min(v_max, v_high);
        break;
      case _Fz_: /* 0-T1*M2, p2/8-p2*2 */
        v_min = 0; v_max = vv[_T1_] * M2;
        v_low = min(v_max, p2/8);
        v_min = max(v_min, v_low);
        v_high = max(v_min, p2*32);
        v_max = min(v_max, v_high);
#ifdef NOTEST
        if (po->is_notest) { v_min = 0; v_max = 0; }
#endif
        break;
      case _FP1_: /* 0-T1/Px1*M2/Py1, p2/8-p2*2 */
        v_min = 0; v_max = vv[_T1_]/vv[_Px1_]*M2/vv[_Py1_];
        v_low = min(v_max, p2/8);
        v_min = max(v_min, v_low);
        v_high = max(v_min, p2*32);
        v_max = min(v_max, v_high);
#ifdef NOTEST
        if (po->is_notest) { v_min = 0; v_max = 0; }
#endif
        break;
      case _Ux1_: /* 1-T1, sqrt(8192/4/Ny - 8192*4/Ny) */
        v_min = 1; v_max = vv[_T1_];
        v_low = min(v_max, (int)sqrt(SUBTILE_SIZE/8/po->Ny));
        v_low = min(v_low, SUBTILE_SIZE/8/po->Ny/M3); /* in case M is large */
        v_min = max(v_min, v_low);
        v_high = max(v_min, (int)sqrt(SUBTILE_SIZE*16/po->Ny));
        v_high = max(v_high, SUBTILE_SIZE*16/po->Ny/M3); /* in case M is small */
        v_max = min(v_max, v_high);
        break;
      case _Uz1_: /* 1-M3, Ux1 */
        v_min = 1; v_max = M3;
        v_low = min(v_max, SUBTILE_SIZE/8/po->Ny/vv[_Ux1_]);
        v_min = max(v_min, v_low);
        v_high = max(v_min, SUBTILE_SIZE*16/po->Ny/vv[_Ux1_]);
        v_max = min(v_max, v_high);
        break;
      case _FU1_: /* 0-T1/Ux1*M3/Uz1, p2/8-p2*2 */
        v_min = 0; v_max = vv[_T1_]/vv[_Ux1_]*M3/vv[_Uz1_];
        v_low = min(v_max, p2/8);
        v_min = max(v_min, v_low);
        v_high = max(v_min, p2*32);
        v_max = min(v_max, v_high);
#ifdef NOTEST
        if (po->is_notest) { v_min = 0; v_max = 0; }
#endif
        break;
      case _Fy1_: /* 0-T1*M3, p2/8-p2*2 */
        v_min = 0; v_max = vv[_T1_]*M3;
        v_low = min(v_max, p2/8);
        v_min = max(v_min, v_low);
        v_high = max(v_min, p2*32);
        v_max = min(v_max, v_high);
#ifdef NOTEST
        if (po->is_notest) { v_min = 0; v_max = 0; }
#endif
        break;
      case _Ry_: /* 0-10, 0-M1 */
        v_min = 0; v_max = 10;
        v_high = max(v_min, M1);
        v_max = min(v_max, v_high);
        if (po->is_oned && vv[_P1_] == 1) { v_min = v_max; }
        else if (po->is_oned && vv[_P1_] == po->p) { v_min = v_max = 0; }
        //v_min = v_max = 2;
        break;
      case _T2_: /* 1-M3, small msg overhead >=1KB (64elem) per node, buffer size overhead <=64MB (4M elems) */
        v_min = 1; v_max = M3;
#if 1
        v_low = min(v_max, 64 * vv[_P1_] / M1 / (M4 * vv[_P1_])); /* small msg overhead >= 64 elems */
        v_min = max(v_min, v_low);
        v_high = max(v_min, (BUFFER_SIZE_LIMIT/8) / M1 / (M4 * vv[_P1_])); /* buffer size overhead <= 4M elems */
        v_max = min(v_max, v_high);
#endif
        break;
      case _W2_: /* 0-10, 0-(M3+T2-1)/T2 */
        if (po->is_W0) {
          v_min = v_max = 0;
        } else {
          v_min = 0; v_max = 5;
          if (M3 == vv[_T2_]) {
            v_min = 0; v_max = 0;
          } else {
            /* 2 * 2 * (M1 * Ny * vv[_T2_]) * (vv[_W2_]+1) <= BUFFER_SIZE_LIMIT */
            v_high = min((M3+vv[_T2_]-1)/vv[_T2_], BUFFER_SIZE_LIMIT / 2 / 2 / (M1 * (M4 * vv[_P1_]) * vv[_T2_]) - 1);
            v_high = max(v_min, v_high);
            v_max = min(v_max, v_high);
          }
        }
        break;
      case _Fy2_: /* 0-T2*M1, p2/8-p2*2 */
        v_min = 0; v_max = vv[_T2_]*M1;
        v_low = min(v_max, vv[_P1_]/8);
        v_min = max(v_min, v_low);
        v_high = max(v_min, vv[_P1_]*8);
        v_max = min(v_max, v_high);
#ifdef NOTEST
        if (po->is_notest) { v_min = 0; v_max = 0; }
#endif
        break;
      case _Pz2_: /* 1-T2, 128-256(8192)-512KB cache, sqrt(8192/2/Ny - 8192*2/Ny) */
        v_min = 1; v_max = vv[_T2_];
        v_low = min(v_max, (int)sqrt(SUBTILE_SIZE/8/po->Ny));
        v_low = min(v_low, SUBTILE_SIZE/8/po->Ny/M1); /* in case M is large */
        v_min = max(v_min, v_low);
        v_high = max(v_min, (int)sqrt(SUBTILE_SIZE*16/po->Ny));
        v_high = max(v_high, SUBTILE_SIZE*16/po->Ny/M1); /* in case M is small */
        v_max = min(v_max, v_high);
        break;
      case _Px2_: /* 1-M1, sqrt(8192/2/Ny) */
        v_min = 1; v_max = M1;
        v_low = min(v_max, SUBTILE_SIZE/8/po->Ny/vv[_Pz2_]);
        v_min = max(v_min, v_low);
        v_high = max(v_min, SUBTILE_SIZE*16/po->Ny/vv[_Pz2_]);
        v_max = min(v_max, v_high);
        break;
      case _FP2_: /* 0-T2/Pz2*M1/Px2, p2/8-p2*2 */
        v_min = 0; v_max = vv[_T2_]/vv[_Pz2_]*M1/vv[_Px2_];
        v_low = min(v_max, vv[_P1_]/8);
        v_min = max(v_min, v_low);
        v_high = max(v_min, vv[_P1_]*32);
        v_max = min(v_max, v_high);
#ifdef NOTEST
        if (po->is_notest) { v_min = 0; v_max = 0; }
#endif
        break;
      case _Uz2_: /* 1-T2, sqrt(8192/4/po->Nx - 8192*4/po->Nx) */
        v_min = 1; v_max = vv[_T2_];
        v_low = min(v_max, (int)sqrt(SUBTILE_SIZE/8/po->Nx));
        v_low = min(v_low, SUBTILE_SIZE/8/po->Nx/M4); /* in case M is large */
        v_min = max(v_min, v_low);
        v_high = max(v_min, (int)sqrt(SUBTILE_SIZE*16/po->Nx));
        v_high = max(v_high, SUBTILE_SIZE*16/po->Nx/M4); /* in case M is small */
        v_max = min(v_max, v_high);
        break;
      case _Uy2_: /* 1-M4, Uz2 */
        v_min = 1; v_max = M4;
        v_low = min(v_max, SUBTILE_SIZE/8/po->Nx/vv[_Uz2_]);
        v_min = max(v_min, v_low);
        v_high = max(v_min, SUBTILE_SIZE*16/po->Nx/vv[_Uz2_]);
        v_max = min(v_max, v_high);
        break;
      case _FU2_: /* 0-T2/Uz2*M4/Uy2, p2/8-p2*2 */
        v_min = 0; v_max = vv[_T2_]/vv[_Uz2_]*M4/vv[_Uy2_];
        v_low = min(v_max, vv[_P1_]/8);
        v_min = max(v_min, v_low);
        v_high = max(v_min, vv[_P1_]*32);
        v_max = min(v_max, v_high);
#ifdef NOTEST
        if (po->is_notest) { v_min = 0; v_max = 0; }
#endif
        break;
      case _Fx_: /* 0-T2*M4, p2/8-p2*2 */
        v_min = 0; v_max = vv[_T2_] * M4;
        v_low = min(v_max, vv[_P1_]/8);
        v_min = max(v_min, v_low);
        v_high = max(v_min, vv[_P1_]*32);
        v_max = min(v_max, v_high);
#ifdef NOTEST
        if (po->is_notest) { v_min = 0; v_max = 0; }
#endif
        break;
      case _V_: /* 0-3 */
        v_min = 0; v_max = 3;
        break;
      case _S_: /* 0, 1 */
        v_min = 0; v_max = 1;
        break;
      }
      g_min = grid_value_ceil(1, v_list, v_list_size, j, v_min);
      g_max = grid_value_floor(1, v_list, v_list_size, j, v_max);
      if (g_min > g_max) g_min = g_max;
      //printf("%s: A i %d j %d v_list_size %d v_min %d v_max %d g_min %d g_max %d x %d vv %d\n", __FUNCTION__, i, j, v_list_size[j], v_min, v_max, g_min, g_max, x[i][j], vv[j]);
      x[i][j] = (rand() % (g_max-g_min+1)) + g_min;
      vv[j] = v_list[j][x[i][j]];
      //printf("%s: i %d j %d v_list_size %d v_min %d v_max %d g_min %d g_max %d x %d vv %d\n", __FUNCTION__, i, j, v_list_size[j], v_min, v_max, g_min, g_max, x[i][j], vv[j]);
      if (j == _P1_) {
        /* make the first two points have p1 = 1 and p1 = p each */
        /* make the third point have p1 = sqrt(p) */
        /* the rests will have random p1 */
        if (i == 0 || i == 1) {
          x[i][j] = g_min;
          vv[j] = v_list[j][x[i][j]];
        } else if (i == 2 || i == 3) {
          x[i][j] = g_max;
          vv[j] = v_list[j][x[i][j]];
        } else if (i == 4 || i == 5) {
          x[i][j] = (g_min + g_max) / 2;
          vv[j] = v_list[j][x[i][j]];
        } else if (i == 6 || i == 7) {
          x[i][j] = (g_min + g_max + 2 - 1) / 2;
          vv[j] = v_list[j][x[i][j]];
        }
        /* TODO: three redundant definitions of lower and upper bound: default, is_infeasible, here */
        p2 = po->p / vv[_P1_];
        /* TODO: redundant definitions of M1-M4 */
        M1 = (po->Nx + vv[_P1_] - 1) / vv[_P1_];
        M2 = (po->Ny + p2 - 1) / p2;
        M3 = (Nz_new + p2 - 1) / p2;
        M4 = (po->Ny + vv[_P1_] - 1) / vv[_P1_];
      }
    } /* end of for j (parameter) */
    if (po->is_oned && vv[_P1_] == 1) {
      vv[_Ry_] = 10;
      x[i][_Ry_] = 10;
#ifdef ADJUST_POINT
#else
      vv[_T2_] = 1; vv[_W2_] = 0;
      vv[_Fy2_] = vv[_FP2_] = vv[_FU2_] = vv[_Fx_] = 0;
      vv[_Pz2_] = vv[_Px2_] = vv[_Uz2_] = vv[_Uy2_] = 1;
      x[i][_T2_] = x[i][_W2_] = 0;
      x[i][_Fy2_] = x[i][_FP2_] = x[i][_FU2_] = x[i][_Fx_] = 0;
      x[i][_Pz2_] = x[i][_Px2_] = x[i][_Uz2_] = x[i][_Uy2_] = 0;
#endif
    }
    if (po->is_oned && vv[_P1_] == po->p) {
      vv[_Ry_] = 0;
      x[i][_Ry_] = 0;
#ifdef ADJUST_POINT
#else
      vv[_T1_] = 1; vv[_W1_] = 0;
      vv[_Fz_] = vv[_FP1_] = vv[_FU1_] = vv[_Fy1_] = 0;
      vv[_Px1_] = vv[_Py1_] = vv[_Ux1_] = vv[_Uz1_] = 1;
      x[i][_T1_] = x[i][_W1_] = 0;
      x[i][_Fz_] = x[i][_FP1_] = x[i][_FU1_] = x[i][_Fy1_] = 0;
      x[i][_Px1_] = x[i][_Py1_] = x[i][_Ux1_] = x[i][_Uz1_] = 0;
#endif
    }
#ifdef ADJUST_POINT
#else
    if (vv[_W1_] == 0) {
      vv[_Fz_] = vv[_FP1_] = vv[_Fy1_] = vv[_FU1_] = 0;
      x[i][_Fz_] = x[i][_FP1_] = x[i][_Fy1_] = x[i][_FU1_] = 0;
    }
    if (vv[_W2_] == 0) {
      vv[_Fy2_] = vv[_FP2_] = vv[_Fx_] = vv[_FU2_] = 0;
      x[i][_Fy2_] = x[i][_FP2_] = x[i][_Fx_] = x[i][_FU2_] = 0;
    }
#endif
  } /* end of for i (point) */
  
  FILE *f;
  f = fopen(po->user_vertex_file, "w");
  for (i = 0; i < PARAM_COUNT + 1; i++) {
    int j;
    for (j = 0; j < PARAM_COUNT; j++) {
      fprintf(f, "%d ", x[i][j]);
    }
    fprintf(f, "\n");
  }
  fclose(f);
}
#endif /* pure random */
#endif /* deterministic */

/* **********************************************************
 @ main routine for auto-tuning
   ********************************************************** */
int ah_tuning(struct _offt_plan *po, double *in, double *out) {
  hsession_t sess;
  hdesc_t *hdesc = NULL;
  const char *name, *retstr;
  int retval;
  int loop_count = 0;
  int every_loop_count = 0;
  name = "fft";
  long ahv[PARAM_COUNT] = {0,};
  int rank = po->rank;
  int exp_num = 0;
  MPI_Datatype MPI_OVERLAP_PARAMS;
  pid_t svr_pid;

  if (rank)
    goto ah_loop;

  /* begin rank0 routine */

  /* setup files for point history and initial vertex */
  srand(time(NULL));
  exp_num = rand() % 100000000;
  sprintf(po->point_database_file, "%s/tmp-db-%08d", HOME_DIR, exp_num);
  sprintf(po->user_vertex_file, "%s/tmp-uv-%08d", HOME_DIR, exp_num);
  //sprintf(po->user_vertex_file, "%s/user-vertex", HOME_DIR);
  printf("point_database_file %s\n", po->point_database_file);
  printf("user_vertex_file %s\n", po->user_vertex_file);

  /* initialize ah client */
  hsession_init(&sess);
  if (hsession_name(&sess, "fft") < 0) {
    fprintf(stderr, "Could not set session name.\n");
    return -1;
  }
  /* define parameter range */
  int *v_list[PARAM_COUNT], v_list_size[PARAM_COUNT];
  params_range_setup(po, v_list, v_list_size);
  int i;
  for (i = 0; i < PARAM_COUNT; i++) {
    char str[4];
    sprintf(str, "V%02d", i);
    hsession_int(&sess, str, 0, v_list_size[i] - 1, 1);
  }
  switch (po->ah_strategy) {
  case 0: hsession_strategy(&sess, "nm.so"); break;
  case 1: hsession_strategy(&sess, "pro.so"); break;
  case 2: hsession_strategy(&sess, "random.so"); break;
  case 3: hsession_strategy(&sess, "brute.so"); break;
  }
  /* set initial simplex */
  hcfg_set(sess.cfg, "SHSONG_USER_VERTEX_FILE", po->user_vertex_file);
  if (po->ah_strategy == 0 || po->ah_strategy == 1) /* only for nm or pro */
    write_initial_simplex(po, v_list, v_list_size);
  retstr = hsession_launch(&sess, NULL, 0);
#if 0 /* launch hserver */
  if (retstr) {
    fprintf(stderr, "Could not launch tuning session: %s\n", retstr);
    return -1;
  }
#else
  if (retstr) {
#if defined(SHSONG_HOPPER)
    char *argv[2] = { "./activeharmony/bin/hserver", NULL };
#elif defined(SHSONG_EDISON)
    char *argv[2] = { "./activeharmony/bin/hserver", NULL };
#else
    char *argv[2] = { "./activeharmony/bin/hserver", NULL };
#endif
    svr_pid = launch_silent(argv[0], argv);
#if defined(SHSONG_HOPPER)
    int itry = 0;
    for (itry = 0; itry < 4; i++) {
      sleep(itry + 1);
      retstr = hsession_launch(&sess, NULL, 0);
      if (!retstr) break;
      fprintf(stderr, "Failed to retry %d hession_launch (%d).\n", itry, svr_pid);
    }
    if (itry == 4) {
      if (svr_pid && kill(svr_pid, SIGKILL) < 0)
        fprintf(stderr, "Could not kill server process (%d).\n", svr_pid);
      return -1;
    }
#else
    sleep(2);
    retstr = hsession_launch(&sess, NULL, 0);
    if (retstr) {
      fprintf(stderr, "Failed to retry hession_launch (%d).\n", svr_pid);
      if (svr_pid && kill(svr_pid, SIGKILL) < 0)
        fprintf(stderr, "Could not kill server process (%d).\n", svr_pid);
      return -1;
    }
#endif
  }
#endif
  printf("Starting Harmony...\n");
  hdesc = harmony_init();
  if (hdesc == NULL) {
    fprintf(stderr, "Failed to initialize a harmony session.\n");
    return -1;
  }
  for (i = 0; i < PARAM_COUNT; i++) {
    char str[4];
    sprintf(str, "V%02d", i);
    harmony_bind_int(hdesc, str, &ahv[i]);
  }
  if (harmony_join(hdesc, NULL, 0, name) < 0) {
    fprintf(stderr, "Could not connect to harmony server: %s\n",
      harmony_error_string(hdesc));
    retval = -1;
    goto cleanup;
  }

  remove(po->point_database_file);

  /* begin tuning routing for all ranks */
ah_loop:
  /* MPI type setting for parameter broadcast */
  //MPI_Datatype MPI_OVERLAP_PARAMS;
  {
    int count = PARAM_COUNT + 3;
    int lengths[count];
    int i;
    MPI_Aint offsets[count];
    MPI_Datatype types[count];
    for (i = 0; i < count; i++) {
      lengths[i] = 1;
      offsets[i] = i * sizeof(int);
      types[i] = MPI_INT;
    }
    MPI_Type_struct(count, lengths, offsets, types, &MPI_OVERLAP_PARAMS);
    MPI_Type_commit(&MPI_OVERLAP_PARAMS);
  }

  loop_count = 0; /* # feasible points */
  every_loop_count = 0; /* total # points */
  while (1) {
    loop_count++;
    every_loop_count++;
    double perf = 0.0;
#ifdef TUNING_REPS
    int r = 0;
#endif

    /* retrieve point and set params */
    if (!rank) {
      po->params->is_converged = 0;
      po->params->is_infeasible = 0;
      po->params->is_in_database = 0;
      /* check convergence */
      if (loop_count >= po->max_loop || every_loop_count >= po->max_loop * 10 || harmony_converged(hdesc)) {
        po->params->is_converged = 1;
        printf("converged loop_count %d\n", loop_count);
      } else {
        /* retrieve point */
        harmony_fetch(hdesc);
        /* convert bwd */
        params_convert(1, po->params->v, ahv, po, v_list, v_list_size);
        print_params(po->params->v);
        /* check infeasibility */
        int err_i;
        po->params->is_infeasible = is_infeasible_point(po, po->params->v, &err_i);
        if (po->params->is_infeasible) {
          perf = 99999999.0;
          printf("INFEASIBLE POINT err_i:%d\n", err_i);
        } else {
          /* check database */
          po->params->is_in_database = is_in_database_point(po, &perf);
          if (po->params->is_in_database)
            printf("%.5f FOUND IN DATABASE\n", perf);
        }
        //printf("%s: c%d i%d d%d\n", __FUNCTION__,
        //po->params->is_converged, po->params->is_infeasible, po->params->is_in_database);
      }
    } /* end of if !rank */

    /* share the retrieved point */
    MPI_Bcast(po->params, 1, MPI_OVERLAP_PARAMS, 0, MPI_COMM_WORLD);
    /* finish if converged */
    if (po->params->is_converged) break;

    /* execute the retrieved point */
    if (po->params->is_infeasible || po->params->is_in_database) {
      loop_count--;
    } else {
      /* setup3: parameter-dependent settings: _offt_comm, buffer, fftw transpose */
      po->comm = offt_comm_malloc(po);
      struct _offt_comm* comm = po->comm;
      po->t_init[INIT_AH] += MPI_Wtime();
      set_buffer(po);
      if (
          !(po->tuning_mode == 1 && po->is_oned && comm->p1 == 1) && 
          !(po->tuning_mode == 2 && po->is_oned && comm->p1 == po->p) && 
          !(po->tuning_mode == 0 && !po->is_oned && po->is_equalxy && comm->M1 == comm->M4) && 
          !(po->tuning_mode == 0 && po->is_oned && comm->p1 != 1 && comm->p1 != po->p && po->is_equalxy && comm->M1 == comm->M4)
      ) {
        if (po->params->v[_S_])
          po->pt_transpose = NULL;
        else
          po->pt_transpose = setup_transpose(po, out);
      } else {
        po->pt_transpose = NULL;
      }
#ifdef P1D_LIST
      setup_p1d(po, out);
#endif
      po->t_init[INIT_AH] -= MPI_Wtime();
#ifdef TUNING_REPS
    perf = 999999999.0;
    for (r = 0; r < TUNING_REPS; r++) {
#endif
      /* initialize input memory */
      {
        int size = (comm->M2*comm->p2 > comm->M4*comm->p1)?
          comm->M1*comm->M2*comm->M3*comm->p2 : comm->M1*comm->M3*comm->M4*comm->p1;
        memset(in, 0, sizeof(double) * size);
      }
      /* execution */
      MPI_Barrier(MPI_COMM_WORLD);
      offt_3d_execute(po, in, out, 1);
#ifdef TUNING_REPS
      if (!rank)
        offt_print_time(po->t);
      perf = (po->t[ALL] < perf)? po->t[ALL]: perf;
    } // end of for r
#endif
      /* free of setup3 except pt_transpose */
#ifdef TRANSPOSE_LIST
#else
      fftw_destroy_plan(po->pt_transpose); po->pt_transpose = NULL;
#endif
      offt_comm_free(po->comm); po->comm = NULL;
      clear_buffer(po);
      //MPI_Reduce(&t, &perf, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
#ifdef TUNING_REPS
#else
      if (!rank)
        offt_print_time(po->t);
      perf = po->t[ALL];
#endif
    }

    /* report the performance */
    if (!rank) {
      //printf("perf = %.5f\n", perf);
      if (!po->params->is_infeasible && !po->params->is_in_database)
        write_to_database(po, perf);
      harmony_report(hdesc, perf);
    }
  } // end of while (1)

  /* retrieve the best point */
  if (!rank) {
    harmony_best(hdesc);
    /* convert bwd */
    params_convert(1, po->params->v, ahv, po, v_list, v_list_size);
  }
  MPI_Bcast(po->params, 1, MPI_OVERLAP_PARAMS, 0, MPI_COMM_WORLD);
  if (!rank) {
    double perf = 0.0;
    printf("@ BEST ");
    print_params(po->params->v);
    if (is_in_database_point(po, &perf)) {
      printf("@ BEST %.5f\n", perf);
    }
  }
  if (!rank && harmony_leave(hdesc) < 0) {
    fprintf(stderr, "Failed to disconnect from harmony server.\n");
    retval = -1;
  }
cleanup:
  if (!rank) {
    harmony_fini(hdesc);
    for (i = 0; i < PARAM_COUNT; i++)
      free(v_list[i]);
    if (svr_pid && kill(svr_pid, SIGKILL) < 0)
      fprintf(stderr, "!Could not kill server process (%d).\n", svr_pid);
  }
  return 0;
}
#endif // AH_TUNING
