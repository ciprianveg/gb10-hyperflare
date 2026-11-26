/* SPDX-License-Identifier: Apache-2.0 */
#ifndef FLARE_ROCE_PEER_ORDER_H
#define FLARE_ROCE_PEER_ORDER_H

/* roce_create validates world/rank; post_op supplies 1 <= step < world.
 * This is a sender-local posting position, not a synchronized collective step. */
static inline int roce_cyclic_peer(int world, int rank, int step) {
  return (rank + step) % world;
}

#endif
