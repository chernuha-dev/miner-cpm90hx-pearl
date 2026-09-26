#ifndef CP_FEE_H
#define CP_FEE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Legacy upstream interface, implemented as zero-fee wallet pass-through. */

void cp_fee_init(const char* user_wallet, int enable);

/* Full-matrix hash-tile count, retained for legacy call sites. */
void cp_fee_set_tiles_per_matrix(uint64_t tiles_per_matrix);

void cp_fee_on_authorized(void);

const char* cp_fee_wallet(void);

void cp_fee_prepare_matrix(void);

int cp_fee_next_is_dev(void);
int cp_fee_needs_switch(void);

void cp_fee_note_tiles(uint64_t tiles);

uint64_t cp_fee_debt(void);
uint64_t cp_fee_tiles_per_matrix(void);
uint64_t cp_fee_threshold(void);
int cp_fee_enabled(void);

#ifdef __cplusplus
}
#endif

#endif /* CP_FEE_H */
