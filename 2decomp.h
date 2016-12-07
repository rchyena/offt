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

#include <stdlib.h>

#ifdef SHSONG_HOPPER
#define DECOMP_FORT_MOD_NAME1(NAME) decomp_2d_##NAME##_
#define DECOMP_FORT_MOD_NAME2(NAME) decomp_2d_fft_##NAME##_
#endif
#ifdef SHSONG_EDISON
#define DECOMP_FORT_MOD_NAME1(NAME) decomp_2d_mp_##NAME##_
#define DECOMP_FORT_MOD_NAME2(NAME) decomp_2d_fft_mp_##NAME##_
//#define DECOMP_FORT_MOD_NAME1(NAME) decomp_2d_##NAME##_
//#define DECOMP_FORT_MOD_NAME2(NAME) decomp_2d_fft_##NAME##_
#endif

//#define DOUBLE_PREC
//#define EVEN
//#define OVERWRITE

#if 0
extern void DECOMP_FORT_MOD_NAME(p3dfft_setup)(int *dims,int *nx,int *ny,int *nz, int *ow, int *memsize);
extern void p3dfft_setup(int *dims,int nx,int ny,int nz,int ovewrite, int *memsize);
inline void p3dfft_setup(int *dims,int nx,int ny,int nz, int overwrite, int * memsize)
{
  DECOMP_FORT_MOD_NAME(p3dfft_setup)(dims,&nx,&ny,&nz,&overwrite,memsize);
}
#endif

#if 0
extern void DECOMP_FORT_MOD_NAME1(decomp_2d_init)(int *nx, int *ny, int *nz, int *p_row, int *p_col);
extern void decomp_2d_init(int *nx, int *ny, int *nz, int *p_row, int *p_col);
inline void decomp_2d_init(int *nx, int *ny, int *nz, int *p_row, int *p_col)
{
  DECOMP_FORT_MOD_NAME1(decomp_2d_init)(nx, ny, nz, p_row, p_col);
}
#else
extern void DECOMP_FORT_MOD_NAME1(decomp_2d_init)(int *nx, int *ny, int *nz, int *p_row, int *p_col, int *periodic_bc);
extern void decomp_2d_init(int nx, int ny, int nz, int p_row, int p_col, int *periodic_bc);
inline void decomp_2d_init(int nx, int ny, int nz, int p_row, int p_col, int *periodic_bc)
{
  DECOMP_FORT_MOD_NAME1(decomp_2d_init)(&nx, &ny, &nz, &p_row, &p_col, periodic_bc);
}
#endif

extern void DECOMP_FORT_MOD_NAME2(fft_init_noarg)();
extern void fft_init_noarg();
inline void fft_init_noarg()
{
  DECOMP_FORT_MOD_NAME2(fft_init_noarg)();
}

extern void DECOMP_FORT_MOD_NAME2(fft_init_arg)(int *pencil);
extern void fft_init_arg(int pencil);
inline void fft_init_arg(int pencil)
{
  DECOMP_FORT_MOD_NAME2(fft_init_arg)(&pencil);
}

#if 0
struct cpx {
  double r;
  double i;
};
extern void DECOMP_FORT_MOD_NAME2(fft_3d_c2c)(struct cpx _in[][4][8], struct cpx _out[][4][8], int *isign);
extern void fft_3d_c2c(double *in, double *out, int isign);
inline void fft_3d_c2c(double *in, double *out, int isign)
{
  DECOMP_FORT_MOD_NAME2(fft_3d_c2c)(&in, &out, &isign);
}
#endif
#if 0
extern void DECOMP_FORT_MOD_NAME2(fft_3d_c2c)(double **in, double **out, int *isign);
extern void fft_3d_c2c(double *in, double *out, int isign);
inline void fft_3d_c2c(double *in, double *out, int isign)
{
  DECOMP_FORT_MOD_NAME2(fft_3d_c2c)(&in, &out, &isign);
}
#endif
#if 1
extern void DECOMP_FORT_MOD_NAME2(fft_3d_c2c)(double *in, double *out, int *isign);
extern void fft_3d_c2c(double *in, double *out, int isign);
inline void fft_3d_c2c(double *in, double *out, int isign)
{
  DECOMP_FORT_MOD_NAME2(fft_3d_c2c)(in, out, &isign);
}
#endif
#if 0
extern void DECOMP_FORT_MOD_NAME2(fft_3d_c2c)(void *in, void *out, int *isign);
extern void fft_3d_c2c(double *in, double *out, int isign);
inline void fft_3d_c2c(double *in, double *out, int isign)
{
  DECOMP_FORT_MOD_NAME2(fft_3d_c2c)((void*)in, (void*)out, &isign);
}
#endif

extern void DECOMP_FORT_MOD_NAME2(decomp_2d_fft_finalize)();
extern void decomp_2d_fft_finalize();
inline void decomp_2d_fft_finalize()
{
  DECOMP_FORT_MOD_NAME2(decomp_2d_fft_finalize)();
}
