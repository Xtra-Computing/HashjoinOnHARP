#ifndef PARALLEL_RADIX_JOIN_H
#define PARALLEL_RADIX_JOIN_H

#include "types.h" /* relation_t */
void *
alloc_aligned(size_t size);
int64_t
RJ(relation_t * relR, relation_t * relS, int nthreads);
void
radix_cluster_nopadding(relation_t * outRel, relation_t * inRel, int R, int D);
int64_t
join_init_run(relation_t * relR, relation_t * relS, tuple_t * tmpRelR, tuple_t * tmpRelS, int nthreads);

#endif
