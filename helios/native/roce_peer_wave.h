#ifndef FLARE_ROCE_PEER_WAVE_H
#define FLARE_ROCE_PEER_WAVE_H

#include <stdint.h>

typedef struct {
  uint32_t chunk;
  uint32_t wave;
  int posted;
} roce_peer_wave_t;

enum {
  ROCE_WAVE_INVALID = -3,
  ROCE_WAVE_POISONED = -2,
  ROCE_WAVE_STOPPED = -1,
  ROCE_WAVE_WAIT = 0,
  ROCE_WAVE_POST = 1,
  ROCE_WAVE_ADVANCED = 2
};

static inline int roce_peer_wave_dest(int rank, int world, uint32_t wave) {
  return (rank + (int)wave) % world;
}

static inline int roce_peer_wave_source(int rank, int world, uint32_t wave) {
  return (rank + world - (int)wave) % world;
}

// Historical host-admission progress only: never authorize payload reads or buffer reuse.
static inline int roce_peer_wave_seen(uint32_t seq, uint32_t seen) {
  if (seq == 0 || seq > INT32_MAX || seen > INT32_MAX) {
    return -1;
  }
  return seen >= seq;
}

// POST does not change state until the caller has successfully posted its peer's stripes.
static inline int roce_peer_wave_step(roce_peer_wave_t* cursor, int world, uint32_t seq,
                                     uint32_t ready, int credit_blocked, int outgoing_done,
                                     int incoming_reached, uint32_t poison, int running) {
  if (poison != 0) {
    return ROCE_WAVE_POISONED;
  }
  if (!running) {
    return ROCE_WAVE_STOPPED;
  }
  if (seq == 0 || seq > INT32_MAX) {
    return ROCE_WAVE_INVALID;
  }
  if (!cursor->posted) {
    return !credit_blocked && ready == seq ? ROCE_WAVE_POST : ROCE_WAVE_WAIT;
  }
  if (incoming_reached < 0) {
    return ROCE_WAVE_INVALID;
  }
  if (!outgoing_done || !incoming_reached) {
    return ROCE_WAVE_WAIT;
  }
  cursor->posted = 0;
  if (++cursor->wave == (uint32_t)world) {
    cursor->wave = 1;
    cursor->chunk++;
  }
  return ROCE_WAVE_ADVANCED;
}

#endif
