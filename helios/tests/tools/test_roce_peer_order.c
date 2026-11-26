/* SPDX-License-Identifier: Apache-2.0 */
#include <assert.h>
#include <stdio.h>
#include "roce_peer_order.h"

int main(void) {
  /* roce_proxy.c supports at most ROCE_MAX_PEERS=16. Exercise every valid
   * geometry, including powers of two and intermediate mesh sizes. */
  for (int world = 2; world <= 16; world++) {
    for (int rank = 0; rank < world; rank++) {
      int seen[16] = {0};
      for (int step = 1; step < world; step++) {
        int peer = roce_cyclic_peer(world, rank, step);
        assert(peer >= 0 && peer < world);
        assert(peer != rank);
        assert(seen[peer]++ == 0);
      }
      for (int peer = 0; peer < world; peer++)
        assert(seen[peer] == (peer != rank));
    }
    for (int step = 1; step < world; step++) {
      int destinations[16] = {0};
      for (int rank = 0; rank < world; rank++)
        destinations[roce_cyclic_peer(world, rank, step)]++;
      for (int peer = 0; peer < world; peer++)
        assert(destinations[peer] == 1);
    }
  }
  const int rank8_peers[] = {9,10,11,12,13,14,15,0,1,2,3,4,5,6,7};
  for (int step = 1; step < 16; step++)
    assert(roce_cyclic_peer(16, 8, step) == rank8_peers[step - 1]);
  puts("PASS: actual post_op peer helper, worlds 2..16, complete nonself coverage, destination permutations, rank8 order");
  return 0;
}
