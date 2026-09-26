#include "cp_fee.h"

#include <string.h>

/* Keep the upstream call sites source-compatible while routing every share to
 * the configured wallet. There is no developer wallet or fee scheduler. */
static char g_user_wallet[256];
static uint64_t g_tiles_per_matrix = 0;

void cp_fee_init(const char* user_wallet, int enable)
{
    (void)enable;
    g_user_wallet[0] = 0;
    if(user_wallet){
        strncpy(g_user_wallet, user_wallet, sizeof(g_user_wallet) - 1);
        g_user_wallet[sizeof(g_user_wallet) - 1] = 0;
    }
    g_tiles_per_matrix = 0;
}

void cp_fee_set_tiles_per_matrix(uint64_t tiles_per_matrix)
{
    g_tiles_per_matrix = tiles_per_matrix;
}

void cp_fee_on_authorized(void) {}
const char* cp_fee_wallet(void) { return g_user_wallet; }
void cp_fee_prepare_matrix(void) {}
int cp_fee_next_is_dev(void) { return 0; }
int cp_fee_needs_switch(void) { return 0; }
void cp_fee_note_tiles(uint64_t tiles) { (void)tiles; }
uint64_t cp_fee_debt(void) { return 0; }
uint64_t cp_fee_tiles_per_matrix(void) { return g_tiles_per_matrix; }
uint64_t cp_fee_threshold(void) { return 0; }
int cp_fee_enabled(void) { return 0; }
