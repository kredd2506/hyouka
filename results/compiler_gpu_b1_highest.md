# Compiler walk — backend=gpu (NVIDIA L4), jax 0.12.1, B=1, precision=highest

## 1. jaxpr: 186 equations
  broadcast_in_dim×34, add×30, mul×17, reduce_sum×16, div×16, sub×16, conv_general_dilated×10, reshape×10, square×8, sqrt×8, integer_pow×8, custom_jvp_call×8, transpose×1, log×1, real×1, imag×1, concatenate×1

## 2. StableHLO: 305 ops
  broadcast_in_dim×102, constant×50, add×46, divide×24, multiply×17, reduce×16, subtract×16, convolution×10, reshape×10, sqrt×8, transpose×1, log×1, real×1, imag×1, concatenate×1, maximum×1

## 3. Optimized HLO: 60 kernels launched per call
  fusion:kInput×49, fusion:kCustom×8, fusion:kLoop×3
  One residual block, in launch order:
    fusion:kInput      f32[1,5,14,128]{3,2,1,0}           input_concatenate_fusion
    fusion:kLoop       f32[128,5,3,3]{3,2,1,0}            wrapped_transpose.8
    fusion:kInput      f32[256]{0}                        input_reduce_fusion.15
    fusion:kInput      f32[]                              input_reduce_fusion.16
    fusion:kInput      f32[256]{0}                        input_reduce_fusion.14
    fusion:kInput      f32[]                              input_reduce_fusion.17
    fusion:kInput      f32[128,14,128]{2,1,0}             wrapped_transpose.7
    fusion:kInput      f32[1,128,14,128]{3,2,1,0}         input_transpose_fusion.1
    fusion:kCustom     f32[128,128,3,3]{3,2,1,0}          wrapped_transpose.9
    fusion:kInput      f32[256]{0}                        input_reduce_fusion.13
    fusion:kInput      f32[]                              input_reduce_fusion.18
    fusion:kInput      f32[256]{0}                        input_reduce_fusion.12
    fusion:kInput      f32[]                              input_reduce_fusion.19

## 4. Memory: args 19.5 MB, output 0.01 MB, temp 5.0 MB
  one activation [B,14,128,128] f32 = 0.92 MB

## 5. Where the time goes (ablation inside the full compiled graph, median ms)
  full NRX                                0.924
  without LayerNorm                       0.843
  without the 8 residual convs            0.374
  without both (in/out conv + glue)       0.236
  dispatch floor (jit x+1)                0.084
  ⇒ 8 residual convs ≈ 0.550 ms (59%), 8 LayerNorms ≈ 0.081 ms (9%), dispatch ≈ 9% of a call

## 6. Pass dump: 563 pass runs, 68 distinct passes, 41 pipelines, in data/xla_dump/gpu/
  pipelines: root → annotate-host-compute → pre-spmd-partitioner → sharding-removal → host-compute → optimization → simplification → simplification-2 → collective-optimizations → conv_canonicalization → reshape_mover_after_conv_canonicalization → simplify_after_conv_canonicalization → layout_assignment → layout_normalization → nvptx_post-layout_assignment_part_1 → hlo_normalization → post-layout_assignment → float_normalization → remove-no-op-reduce-precision-algebraic-simplifier → nvptx_post-layout_assignment_part_2 → dynamic-slice → pre-fusion → fusion → post-fusion_optimization → async-collective-conversion → post-fusion-simplification-pipeline_optimization → post-fusion-verification-pipeline_optimization → collective-schedule-linearizer → invariant-checkers → async-wrapper → fusion-wrapper → config-assigner → propagate-call-metadata → pre-scheduling-copy-insertion → pre-scheduling-passes → scheduled-gpu-module → post-scheduler-xla-transforms → post-scheduling-passes → async-to-sync-converter → remat-pipeline → sanitize-constant-names
  most frequent: pipeline-end×52, gpu-algebraic-simplifier×36, float-normalization-bf16×33, fusion-wrapper×33, convert-mover×22, zero_sized_hlo_elimination×16, dce×13, sub-byte-size-setter×12
  first 8: annotate-host-compute → hlo_host_device_type_call_wrapper → erase-xla-late-frontend-attributes → sharding-remover → host_compute_asyncifier → ragged_dot_rewriter → zero_sized_hlo_elimination → convert-mover
  last 8:  apply-xla-transforms-120 → async-to-sync-converter → gpu-convert-async-collectives-to-sync → rematerialization → fusion-wrapper → sanitize-constant-names → pipeline-end → pipeline-end
