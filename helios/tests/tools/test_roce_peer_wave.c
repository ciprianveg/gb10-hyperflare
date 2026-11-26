// CPU-only admission-state tests; no verbs endpoint or GPU is opened.
#include "../../native/roce_proxy.c"
#include <assert.h>

enum { WORLD_LIMIT = 3, CHUNK_LIMIT = 5 };

static roce_ctx_t* context(int world, int rank) {
  roce_ctx_t* c = calloc(1, sizeof(*c));
  assert(c);
  c->world = world;
  c->rank = rank;
  c->n_rail = 2;
  c->n_lane = 4;
  c->n_hca = 2;
  c->pipe_flag_off = 128;
  c->region = calloc(1, 128 + 2 * world * ROCE_MAX_CHUNKS * 2 * ROCE_FLAG_STRIDE);
  assert(c->region);
  for (int p = 0; p < world; p++) {
    c->rail_hca[p][0] = 0;
    c->rail_hca[p][1] = 1;
  }
  return c;
}

static void release_context(roce_ctx_t* c) {
  free(c->region);
  free(c);
}

static void flag(roce_ctx_t* c, int phase, int source, int chunk, int rail, uint32_t value) {
  size_t offset = c->pipe_flag_off +
      ((phase * c->world + source) * ROCE_MAX_CHUNKS * c->n_rail +
       chunk * c->n_rail + rail) * ROCE_FLAG_STRIDE;
  __atomic_store_n((uint32_t*)(c->region + offset), value, __ATOMIC_RELEASE);
}

static void flags_and_completion(void) {
  roce_ctx_t* c = context(3, 1);
  for (int phase = 0; phase < 2; phase++) {
    for (int chunk = 0; chunk < 5; chunk++) {
      flag(c, phase, 2, chunk, 0, 6);
      flag(c, phase, 2, chunk, 1, 0);
      assert(pipe_peer_chunk_reached(c, 7, phase, 2, chunk) == 0);
      flag(c, phase, 2, chunk, 0, 7);
      assert(pipe_peer_chunk_reached(c, 7, phase, 2, chunk) == 0);
      flag(c, phase, 2, chunk, 1, 8);
      assert(pipe_peer_chunk_reached(c, 7, phase, 2, chunk) == 1);
      flag(c, phase, 2, chunk, 1, UINT32_MAX);
      assert(pipe_peer_chunk_reached(c, 7, phase, 2, chunk) == -1);
      flag(c, phase, 2, chunk, 1, INT32_MAX);
      flag(c, phase, 2, chunk, 0, INT32_MAX);
      assert(pipe_peer_chunk_reached(c, INT32_MAX, phase, 2, chunk) == 1);
      assert(pipe_peer_chunk_reached(c, 0, phase, 2, chunk) == -1);
      assert(pipe_peer_chunk_reached(c, UINT32_MAX, phase, 2, chunk) == -1);
      assert(pipe_peer_chunk_reached(c, 7, phase, 0, chunk) == 0);
      int lane = chunk % c->n_lane;
      c->hca[0].outstanding[2][lane] = 1;
      assert(!pipe_peer_chunk_completed(c, 2, chunk));
      c->hca[0].outstanding[2][lane] = 0;
      c->hca[1].outstanding[2][lane] = 1;
      assert(!pipe_peer_chunk_completed(c, 2, chunk));
      c->hca[1].outstanding[2][lane] = 0;
      c->hca[0].outstanding[0][lane] = 9;
      c->hca[0].outstanding[2][(lane + 1) % c->n_lane] = 9;
      assert(pipe_peer_chunk_completed(c, 2, chunk));
      c->hca[0].outstanding[0][lane] = 0;
      c->hca[0].outstanding[2][(lane + 1) % c->n_lane] = 0;
      // The production completion helper has no generation/wave-dependent QP mapping.
      for (uint32_t seq = 1; seq <= 3; seq++) {
        flag(c, phase, 2, chunk, 0, seq);
        flag(c, phase, 2, chunk, 1, seq);
        c->hca[0].outstanding[2][lane] = 1;
        assert(!pipe_peer_chunk_completed(c, 2, chunk));
        c->hca[0].outstanding[2][lane] = 0;
        assert(pipe_peer_chunk_completed(c, 2, chunk));
      }
    }
  }
  release_context(c);
}

static void cursor_checks(void) {
  roce_peer_wave_t c = {.wave = 1};
  assert(roce_peer_wave_step(&c, 3, 7, 8, 0, 1, 1, 0, 1) == ROCE_WAVE_WAIT);
  assert(roce_peer_wave_step(&c, 3, 7, 7, 1, 1, 1, 0, 1) == ROCE_WAVE_WAIT);
  assert(roce_peer_wave_step(&c, 3, 7, 7, 0, 0, 0, 0, 1) == ROCE_WAVE_POST);
  assert(!c.posted && c.chunk == 0 && c.wave == 1);
  c.posted = 1;  // successful posting, as in run_pipe_op
  assert(roce_peer_wave_step(&c, 3, 7, 7, 0, 0, 1, 0, 1) == ROCE_WAVE_WAIT);
  assert(roce_peer_wave_step(&c, 3, 7, 7, 0, 1, 0, 0, 1) == ROCE_WAVE_WAIT);
  assert(roce_peer_wave_step(&c, 3, 7, 7, 0, 1, -1, 0, 1) == ROCE_WAVE_INVALID);
  assert(roce_peer_wave_step(&c, 3, 7, 7, 0, 0, 0, 7, 1) == ROCE_WAVE_POISONED);
  assert(roce_peer_wave_step(&c, 3, 7, 7, 0, 0, 0, 0, 0) == ROCE_WAVE_STOPPED);
  assert(c.posted && c.chunk == 0 && c.wave == 1);
  assert(roce_peer_wave_step(&c, 3, 7, 7, 0, 1, 1, 0, 1) == ROCE_WAVE_ADVANCED);
  assert(!c.posted && c.wave == 2 && c.chunk == 0);
  c.posted = 1;
  assert(roce_peer_wave_step(&c, 3, 7, 7, 0, 1, 1, 0, 1) == ROCE_WAVE_ADVANCED);
  assert(!c.posted && c.wave == 1 && c.chunk == 1);
  assert(roce_peer_wave_step(&c, 3, 7, 7, 0, 0, 0, 7, 1) == ROCE_WAVE_POISONED);
  assert(roce_peer_wave_step(&c, 3, 7, 7, 0, 0, 0, 0, 0) == ROCE_WAVE_STOPPED);
}

typedef struct {
  int posted, delivered[2], completed[2], arrival_at[2], completion_at[2];
} transfer_t;

// Enumerated delay/order schedules model reliable, per-QP ordered delivery, not RDMA hardware.
static int simulate(int world, int chunks, const int* order, unsigned delays, int ag_first,
                    int drop_one_flag) {
  roce_ctx_t* c[WORLD_LIMIT];
  roce_peer_wave_t cursors[WORLD_LIMIT][2] = {0};
  transfer_t tx[2][WORLD_LIMIT][CHUNK_LIMIT][WORLD_LIMIT] = {0};
  int last_arrival[WORLD_LIMIT][WORLD_LIMIT][4][2] = {0};
  int last_completion[WORLD_LIMIT][WORLD_LIMIT][4][2] = {0};
  int ag_ready_at[WORLD_LIMIT][CHUNK_LIMIT] = {0};
  int posts = 0, advances = 0, finished = 0;
  for (int r = 0; r < world; r++) {
    c[r] = context(world, r);
    cursors[r][0].wave = cursors[r][1].wave = 1;
  }
  for (int tick = 1; tick < 400 && !finished; tick++) {
    for (int phase = 0; phase < 2; phase++)
      for (int src = 0; src < world; src++)
        for (int chunk = 0; chunk < chunks; chunk++)
          for (int dest = 0; dest < world; dest++) {
            transfer_t* t = &tx[phase][src][chunk][dest];
            if (!t->posted) continue;
            for (int rail = 0; rail < 2; rail++) {
              if (!t->delivered[rail] && tick >= t->arrival_at[rail]) {
                if (!(drop_one_flag && phase == 0 && src == 0 && chunk == 0 &&
                      dest == 1 && rail == 1)) {
                  flag(c[dest], phase, src, chunk, rail, 7);
                }
                t->delivered[rail] = 1;
              }
              if (!t->completed[rail] && tick >= t->completion_at[rail]) {
                uint32_t* pending = &c[src]->hca[rail].outstanding[dest][chunk % 4];
                assert(*pending > 0);
                --*pending;
                t->completed[rail] = 1;
              }
            }
          }
    for (int oi = 0; oi < world; oi++) {
      int r = order[oi];
      for (int pi = 0; pi < 2; pi++) {
        int phase = ag_first ? 1 - pi : pi;
        roce_peer_wave_t* cursor = &cursors[r][phase];
        if (cursor->chunk == (uint32_t)chunks) continue;
        int chunk = (int)cursor->chunk;
        int dest = roce_peer_wave_dest(r, world, cursor->wave);
        int src = roce_peer_wave_source(r, world, cursor->wave);
        assert(dest != r && src != r);
        assert(roce_peer_wave_dest(src, world, cursor->wave) == r);
        int rs_ready_at = 1 + ((delays & 1) ? 2 * r : 0) +
                          ((delays & 32) ? chunk : 0);
        int ready = tick >= rs_ready_at;
        if (phase == 1) {
          // The captured GPU stages all RS chunks before reducing and publishing AG.
          ready = tick >= 1 + ((delays & 1) ? 2 * r : 0) +
                          ((delays & 32) ? chunks - 1 : 0);
          for (int peer = 0; peer < world; peer++)
            if (peer != r && pipe_peer_chunk_reached(c[r], 7, 0, peer, chunk) != 1)
              ready = 0;
          if (ready && !ag_ready_at[r][chunk])
            ag_ready_at[r][chunk] = tick + ((delays & 2) ? 5 + (r + chunk) % 3 : 0);
          ready = ready && tick >= ag_ready_at[r][chunk];
        }
        int incoming = cursor->posted ?
            pipe_peer_chunk_reached(c[r], 7, phase, src, chunk) : 0;
        int outgoing = cursor->posted ? pipe_peer_chunk_completed(c[r], dest, chunk) : 0;
        int action = roce_peer_wave_step(cursor, world, 7, ready ? 7 : 0,
            (delays & 16) && tick < 7 + r + phase, outgoing, incoming, 0, 1);
        assert(action >= 0);
        if (action == ROCE_WAVE_POST) {
          transfer_t* t = &tx[phase][r][chunk][dest];
          assert(!t->posted);  // duplicate posts are not masked
          t->posted = 1;
          for (int rail = 0; rail < 2; rail++) {
            int lane = chunk % 4;
            int arrive = tick + 1 + ((delays & 4) ? (rail + 1) * 3 + r : 0);
            if (arrive <= last_arrival[r][dest][lane][rail])
              arrive = last_arrival[r][dest][lane][rail] + 1;
            int complete = arrive + ((delays & 8) ? 8 + (dest + rail) % 3 : 0);
            if (complete <= last_completion[r][dest][lane][rail])
              complete = last_completion[r][dest][lane][rail] + 1;
            last_arrival[r][dest][lane][rail] = t->arrival_at[rail] = arrive;
            last_completion[r][dest][lane][rail] = t->completion_at[rail] = complete;
            c[r]->hca[rail].outstanding[dest][lane]++;
          }
          cursor->posted = 1;
          posts++;
        } else if (action == ROCE_WAVE_ADVANCED) {
          assert(tx[phase][r][chunk][dest].posted);  // post precedes wait/advance
          advances++;
        }
      }
    }
    finished = 1;
    for (int r = 0; r < world; r++)
      for (int phase = 0; phase < 2; phase++)
        if (cursors[r][phase].chunk != (uint32_t)chunks) finished = 0;
  }
  if (finished) {
    assert(!drop_one_flag);
    assert(posts == 2 * world * (world - 1) * chunks && advances == posts);
    for (int phase = 0; phase < 2; phase++)
      for (int src = 0; src < world; src++)
        for (int chunk = 0; chunk < chunks; chunk++)
          for (int dest = 0; dest < world; dest++)
            if (dest != src) {
              transfer_t* t = &tx[phase][src][chunk][dest];
              assert(t->posted && t->delivered[0] && t->delivered[1] &&
                     t->completed[0] && t->completed[1]);
            }
  }
  for (int r = 0; r < world; r++) release_context(c[r]);
  return finished;
}

int main(void) {
  flags_and_completion();
  cursor_checks();
  const int orders[][3] = {{0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}};
  const int sizes[] = {1,2,5};
  unsigned cases = 0;
  for (int world = 2; world <= 3; world++)
    for (int oi = 0; oi < 6; oi++) {
      if (world == 2 && oi != 0 && oi != 2) continue;
      for (unsigned mask = 0; mask < 64; mask++)
        for (int first = 0; first < 2; first++)
          for (int size = 0; size < 3; size++) {
            assert(simulate(world, sizes[size], orders[oi], mask, first, 0));
            cases++;
          }
    }
  assert(!simulate(3, 2, orders[0], 63, 1, 1));
  printf("PASS: flags/completion/cursor guards; %u exhaustive bounded delay/order cases; "
         "missing-rail transfer stalls without premature finish.\n", cases);
  return 0;
}
