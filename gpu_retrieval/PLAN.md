# GPU Retrieval Pipeline with Parallel Preprocessing and Top-k Selection

Parallel Computing course project plan.

## 1. Problem
Given a corpus of N vectors (dim D) and Q query vectors, return for each query the
k most similar corpus vectors (cosine similarity). This is the core of dense retrieval
(RAG, semantic search, recommendation). The cost is dominated by Q x N x D multiply-adds
plus selecting top-k from Q x N scores, so it is a good fit for data parallelism.

## 2. Pipeline stages
| Stage | What | Parallelism |
|---|---|---|
| P1 Preprocess | L2-normalize every corpus and query vector | one CUDA block per vector, shared-memory reduction (CPU: OpenMP over rows) |
| P2 Score | scores[q][n] = dot(query_q, corpus_n) | 32x32 shared-memory tiled kernel, coalesced loads (CPU: OpenMP over queries) |
| P3 Top-k | k best scores per query | see variants below |
| P4 Transfer | H2D of data, D2H of k results per query | measured separately to show the bottleneck |

### Top-k variants (the main parallel-computing contribution)
- **V0 CPU serial**: baseline, insertion into a size-k sorted list.
- **V1 CPU OpenMP**: parallel over queries.
- **V2 GPU Thrust sort**: full `sort_by_key` per query, then take first k. Simple, wasteful O(N log N).
- **V3 GPU two-stage block top-k (ours)**: stage 1: B blocks per query, each thread keeps a
  private sorted list of k, block merges the thread lists with k rounds of block-wide argmax
  (only k values leave each block). Stage 2: one block per query merges B*k candidates with the same kernel.
  O(N) work, no full sort, no atomics.

## 3. Milestones
1. Setup, data generator, CPU serial and OpenMP baselines (done first, testable anywhere).
2. CUDA preprocessing and scoring kernels.
3. CUDA top-k variants V2 and V3.
4. Correctness harness: recall@k and max score difference versus CPU reference.
5. Benchmark sweeps: N, D, k, Q. Stage time breakdown. Speedup tables and plots.
6. Profiling with Nsight Compute/Systems (occupancy, coalescing, shared-memory use).
7. Report and presentation.

## 4. Experiments for the report
- Speedup versus serial CPU and versus OpenMP CPU
- Stage breakdown (transfer vs preprocess vs score vs top-k)
- Top-k variant comparison versus k in {1, 5, 10, 16, 32} and N in {2^16 ... 2^22}
- Scaling with block count B per query
- Correctness: recall@k = 1.0 expected, max |score diff| around 1e-6
- Discussion: memory-bound vs compute-bound, why transfer can dominate, Amdahl's law on the serial part

## 5. Known limits and extensions
- k <= 32 (per-thread registers/local memory). Extension: radix-select for large k.
- Scores matrix Q x N is materialized (processed in query batches). Extension: fuse scoring and top-k.
- Extension: cuBLAS SGEMM for scoring, FP16/tensor cores, multi-stream overlap of transfer and compute.

## 6. Layout
```
src/common.h        config, data generator, timer, verification
src/cpu_pipeline.*  serial + OpenMP implementation
src/main_cpu.cpp    CPU-only benchmark
src/gpu_pipeline.cu CUDA kernels and host pipeline
src/main_gpu.cu     GPU benchmark vs CPU reference
Makefile            make cpu | make gpu
scripts/sweep.sh    benchmark sweeps to CSV
```
