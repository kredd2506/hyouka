# Compiler walk — backend=cpu, jax 0.12.1, B=1

## 1. jaxpr: 186 equations
  broadcast_in_dim×34, add×30, mul×17, reduce_sum×16, div×16, sub×16, conv_general_dilated×10, reshape×10, square×8, sqrt×8, integer_pow×8, custom_jvp_call×8, transpose×1, log×1, real×1, imag×1, concatenate×1

## 2. StableHLO: 305 ops
  broadcast_in_dim×102, constant×50, add×46, divide×24, multiply×17, reduce×16, subtract×16, convolution×10, reshape×10, sqrt×8, transpose×1, log×1, real×1, imag×1, concatenate×1, maximum×1

## 3. Optimized HLO: 62 kernels launched per call
  fusion:kLoop×36, fusion:kCustom×26
  One residual block, in launch order:
    fusion:kLoop       f32[1,14,128,128]{3,2,1,0}         broadcast_subtract_fusion.7
    fusion:kCustom     f32[1]{0}                          ynn_fusion.23
    fusion:kLoop       f32[1]{0}                          add_rsqrt_fusion.7
    fusion:kLoop       f32[1,14,128,128]{3,2,1,0}         broadcast_maximum_fusion
    fusion:kCustom     f32[1,14,128,128]{3,2,1,0}         ynn_fusion.22
    fusion:kLoop       f32[1,14,128,128]{3,2,1,0}         broadcast_add_fusion.4
    fusion:kCustom     f32[1]{0}                          ynn_fusion.21
    fusion:kLoop       f32[1,14,128,128]{3,2,1,0}         broadcast_subtract_fusion.6
    fusion:kCustom     f32[1]{0}                          ynn_fusion.20
    fusion:kLoop       f32[1]{0}                          add_rsqrt_fusion.6
    fusion:kLoop       f32[1,14,128,128]{3,2,1,0}         add_maximum_fusion.6
    fusion:kCustom     f32[1,14,128,128]{3,2,1,0}         ynn_fusion.19
    fusion:kLoop       f32[1,14,128,128]{3,2,1,0}         add_add_fusion.3
    fusion:kCustom     f32[1]{0}                          ynn_fusion.18
    fusion:kLoop       f32[1,14,128,128]{3,2,1,0}         broadcast_subtract_fusion.5
    fusion:kCustom     f32[1]{0}                          ynn_fusion.17

## 4. Memory: args 19.5 MB, output 0.01 MB, temp 2.8 MB
  one activation [B,14,128,128] f32 = 0.92 MB

## 5. Where the time goes (isolated microbenchmarks, median ms)
  conv 128→128 3×3            0.271
  conv + bias                 0.281   (bias add costs +0.010)
  LayerNorm                   0.079
  LayerNorm + ReLU            0.080
  full NRX                    3.013
  8×(conv+bias) = 2.251 (75%),  8×(LN+ReLU) = 0.637 (21%),  sum 2.888 vs full 3.013

## 6. Pass dump: 202 pass runs, 35 distinct passes, 14 pipelines, in data/xla_dump/cpu/
  pipelines: async-collective → pre-spmd-partitioner → sharding-removal → SubbytePacker_pipeline → HLO_passes_through_layout_assignment → simplification → embedded_scatter_expansion → post_scatter_expansion_simplification → hlo_normalization → dot-library-passes → HLO_passes_after_layout_assignment → after_layout_assignment → simplification_after_layout_assignment → HLO_passes_after_scheduling
  most frequent: algsimp×37, sub-byte-size-setter×22, pipeline-end×12, async-collective-custom-call-rewriter×11, strip-memory-placement-annotations×11, flatten-call-graph×11, batched_gather_scatter_normalizer×11, scatter_expander×11
  first 8: async-collective-custom-call-rewriter → strip-memory-placement-annotations → flatten-call-graph → sub-byte-size-setter → batched_gather_scatter_normalizer → algsimp → scatter_expander → algsimp
  last 8:  after_layout_assignment → pipeline-end → fusion-wrapper → cpu-all-reduce-combiner → algsimp → dce → small-while-loop-hoisting → apply-xla-transforms-21
