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

#ifndef OFFT_INTERNAL_INCLUDE
#define OFFT_INTERNAL_INCLUDE

#include "offt.h"

struct _offt_comm* offt_comm_malloc(struct _offt_plan* po);
void offt_comm_free(struct _offt_comm *comm);
void set_buffer(struct _offt_plan *po);
void clear_buffer(struct _offt_plan *po);
#ifdef AH_TUNING
fftw_plan setup_transpose(struct _offt_plan* po, double* out);
#ifdef P1D_LIST
void setup_p1d(struct _offt_plan* po, double* out);
#endif
int grid_value_floor(int is_index, int **v_list, int *v_list_size, int i, int raw_v);
int grid_value_ceil(int is_index, int **v_list, int *v_list_size, int i, int raw_v);
void params_range_setup(struct _offt_plan *po, int** v_list, int* v_list_size);
int ah_tuning(struct _offt_plan *po, double *in, double *out);
#endif // AH_TUNING

#endif // OFFT_INTERNAL_INCLUDE
