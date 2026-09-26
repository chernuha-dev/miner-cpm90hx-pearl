#ifndef CP_AMPERE_TRANSCRIPT_CUH
#define CP_AMPERE_TRANSCRIPT_CUH

/* Hash one 8x16 tile transcript emitted by kernels/ampere_scan.py. */
__global__ void cp_ampere_transcript_jackpot_kernel(
    const uint32_t* __restrict__ transcript,
    int row_batch_count, int col_batch_count,
    int row_period0, int col_period0,
    uint32_t b0, uint32_t b1, uint32_t b2, uint32_t b3,
    uint32_t b4, uint32_t b5, uint32_t b6, uint32_t b7,
    const uint32_t* __restrict__ a_key8,
    int* __restrict__ out_t_rows,
    int* __restrict__ out_t_cols,
    int* __restrict__ found_flag,
    int register_layout)
{
    const int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int tiles_per_period = 256;
    if(idx >= row_batch_count * col_batch_count * tiles_per_period) return;
    if(*found_flag) return;

    const int row_in_batch = idx / (col_batch_count * tiles_per_period);
    const int rem = idx % (col_batch_count * tiles_per_period);
    const int col_in_batch = rem / tiles_per_period;
    const int tile = rem % tiles_per_period;
    const int half = tile / 128;
    const int tile_in_half = tile % 128;
    const int row_tile = tile_in_half / 8;
    const int col_tile = half * 8 + tile_in_half % 8;
    const int cta = row_in_batch * (col_batch_count * 2)
                  + col_in_batch * 2 + half;

    const int warp = (col_tile / 4) * 2 + row_tile / 8;
    const int lane = (row_tile % 8) * 4 + col_tile % 4;
    const int register_thread = warp * 32 + lane;
    uint32_t msg[16];
    #pragma unroll
    for(int step = 0; step < 16; step++)
        msg[step] = register_layout
            ? transcript[(((size_t)row_in_batch * col_batch_count + col_in_batch)
                          * 16 + step) * 256 + register_thread]
            : transcript[((size_t)cta * 16 + step) * 128 + tile_in_half];

    uint32_t digest[8];
    b3_compress64(a_key8, msg, digest);
    const uint32_t tgt[8] = {b0, b1, b2, b3, b4, b5, b6, b7};
    bool ok = true;
    for(int w = 7; w >= 0; w--){
        if(digest[w] < tgt[w]) break;
        if(digest[w] > tgt[w]){ ok = false; break; }
    }
    if(ok && atomicCAS(found_flag, 0, 1) == 0){
        *out_t_rows = (row_period0 + row_in_batch) * PP_ROW_PERIOD
                    + pp_period_row_base(row_tile);
        *out_t_cols = (col_period0 + col_in_batch) * PP_COL_PERIOD
                    + pp_period_col_base(col_tile);
    }
}

#endif
