# Compiler walk — backend=tpu (TPU v5 lite), jax 0.12.1, B=1, precision=high

## 1. jaxpr: 186 equations
  broadcast_in_dim×34, add×30, mul×17, reduce_sum×16, div×16, sub×16, conv_general_dilated×10, reshape×10, square×8, sqrt×8, integer_pow×8, custom_jvp_call×8, transpose×1, log×1, real×1, imag×1, concatenate×1

## 2. StableHLO: 305 ops
  broadcast_in_dim×102, constant×50, add×46, divide×24, multiply×17, reduce×16, subtract×16, convolution×10, reshape×10, sqrt×8, transpose×1, log×1, real×1, imag×1, concatenate×1, maximum×1

## 3. Optimized HLO: 113 top-level ops per call (each one a kernel launch on GPU)
  fusion:kLoop×27, copy×19, multiply×17, pad×10, fusion:kOutput×10, slice×9, add×8, rsqrt×8, custom-call×2, reshape×2, log×1
  One residual block, in launch order:
    pad                f32[5]{0:T(128)S(1)}               pad.43.clone
    custom-call        f32[1,2,14,128]{3,1,2,0:T(2,128)S(1)} custom-call.1
    custom-call        f32[1,2,14,128]{3,1,2,0:T(2,128)S(1)} custom-call
    fusion:kLoop       f32[14,1,128,5]{2,3,0,1:T(8,128)S(1)} maximum_bitcast_fusion
    pad                f32[14,1,136,5]{2,3,0,1:T(8,128)S(1)} pad
    reshape            f32[14,1,8,17,5]{3,2,1,4,0:T(8,128)S(1)} reshape.245
    copy               f32[14,1,8,17,5]{4,2,3,0,1:T(8,128)S(1)} copy.107
    slice              f32[14,1,7,2,5]{4,2,3,0,1:T(8,128)S(1)} slice
    fusion:kOutput     (f32[]{:T(128)}, f32[14,8,17,128]{3,1,2,0:T(8,128)S(1)}) select_reduce_fusion.15
    multiply           f32[]{:T(128)S(6)}                 multiply.24
    fusion:kLoop       f32[]{:T(128)}                     select_reduce_fusion.14
    multiply           f32[]{:T(128)S(6)}                 multiply.25
    add                f32[]{:T(128)S(6)}                 add.120
    rsqrt              f32[]{:T(128)S(6)}                 rsqrt.8
    fusion:kLoop       f32[14,8,17,128]{3,1,2,0:T(8,128)S(1)} subtract_multiply_fusion.7
    copy               f32[14,8,17,128]{3,0,2,1:T(8,128)S(1)} copy.109

## 4. Memory: args 19.5 MB, output 0.01 MB, temp 0.8 MB
  one activation [B,14,128,128] f32 = 0.92 MB

## 5. Where the time goes (ablation inside the full compiled graph, median ms)
  full NRX                                0.416
  without LayerNorm                       0.349
  without the 8 residual convs            0.260
  without both (in/out conv + glue)       0.260
  dispatch floor (jit x+1)                0.171
  ⇒ 8 residual convs ≈ 0.155 ms (37%), 8 LayerNorms ≈ 0.067 ms (16%), dispatch ≈ 41% of a call

## 6. Pass dump: 602 pass runs, 74 distinct passes, 42 pipelines, in data/xla_dump/tpu/
  pipelines: hlo_device_type_async_wrapper → Process_DebugLog_custom_call → Before_X64_rewriter → X64_elimination → Phase_1_pre-layout_assignment_passes → convolution_folding → simplification-1 → Phase_2_pre-layout_assignment_passes → simplification-2 → simplification-3 → post-space-to-batch-simplification → simplification-4 → simplification-4-spmd → Pre_layout_assignment → while-loop-simplification → Pre_layout_assignment_async_collective_custom_call_rewriter → Layout_assignment → JF_dot_canonicalization → Post_layout_assignment → JF_after_layout_assignment → Pre_main_fusion → layout_fixup → host_offloading → while_loop_async_pipelining → Main_fusion → JF_after_main_fusion → Post_main_fusion → multi_output_fusion_and_bf16_folding → while_loop_async_pipelining_post_fusion_scheduling → After_rewrite_shapes → Global_Copy_Insertion → Post_Fusion_Tiling_Assignment → SparseCore_Manual_Offload → Float_type_correction_after_SC_offloading → Post-Global_Copy_Insertion → final_scheduler → apply_xla_transforms_pre_scheduling → async_scheduling → final_rematerialization → Pre_memory_space_assignment → HT_info_assignment → Rename_TensorCore_Fusions
  most frequent: simplify-sorts×49, cse×35, pipeline-end×24, tpu-copy-normalizer×22, algsimp×15, tuple-simplifier×14, tpu_fusion×12, while-initializer-removal×12
  first 8: add-random-host-offloading → debug-log-custom-call-processor → propagate-deep-pipeline-scheduling-mode → bitcast_dtypes_expander → zero_sized_hlo_elimination → convolution-folding → cse → simplification-2
  last 8:  latency-hiding-scheduler → async-collective-fusion → rematerialization_with_stats → root-instruction-sinker → plumb-vmem-info → early-scalar-operand-scheduler → host-transfer-channel-id-assigner → rename-tensor-core-fusion-ops
