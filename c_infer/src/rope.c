#include "rope.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

int mt_rope_init(mt_rope_t *r, int max_pos, int dh, float base) {
  r->max_pos = max_pos;
  r->dh = dh;
  int half = dh / 2;
  r->cos_t = (float *)malloc((size_t)max_pos * half * sizeof(float));
  r->sin_t = (float *)malloc((size_t)max_pos * half * sizeof(float));
  if (!r->cos_t || !r->sin_t) return -1;
  /* inv_freq[p] = 1 / base^(2p/dh) ；freq = pos * inv_freq */
  for (int p = 0; p < half; p++) {
    double inv = 1.0 / pow((double)base, (2.0 * p) / (double)dh);
    for (int pos = 0; pos < max_pos; pos++) {
      double ang = (double)pos * inv;
      r->cos_t[(size_t)pos * half + p] = (float)cos(ang);
      r->sin_t[(size_t)pos * half + p] = (float)sin(ang);
    }
  }
  return 0;
}

void mt_rope_free(mt_rope_t *r) { free(r->cos_t); free(r->sin_t); r->cos_t = r->sin_t = NULL; }

void mt_rope_apply(const mt_rope_t *r, float *x, int B, int H, int L, int offset) {
  int half = r->dh / 2;
  for (int b = 0; b < B; b++)
    for (int h = 0; h < H; h++)
      for (int l = 0; l < L; l++) {
        float *v = x + (((size_t)b * H + h) * L + l) * r->dh;
        const float *c = r->cos_t + (size_t)(offset + l) * half;
        const float *s = r->sin_t + (size_t)(offset + l) * half;
        for (int p = 0; p < half; p++) {
          float a = v[2 * p], bb = v[2 * p + 1];
          v[2 * p]     = a * c[p] - bb * s[p];
          v[2 * p + 1] = a * s[p] + bb * c[p];
        }
      }
}
