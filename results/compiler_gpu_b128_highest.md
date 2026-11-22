# Compiler walk — backend=gpu (NVIDIA L4), jax 0.12.1, B=128, precision=highest

## 1. jaxpr: 186 equations
  broadcast_in_dim×34, add×30, mul×17, reduce_sum×16, div×16, sub×16, conv_general_dilated×10, reshape×10, square×8, sqrt×8, integer_pow×8, custom_jvp_call×8, transpose×1, log×1, real×1, imag×1, concatenate×1

## 2. StableHLO: 321 ops
  broadcast_in_dim×118, constant×50, add×46, divide×24, multiply×17, reduce×16, subtract×16, convolution×10, reshape×10, sqrt×8, transpose×1, log×1, real×1, imag×1, concatenate×1, maximum×1

## 3. Optimized HLO: 85 kernels launched per call
  fusion:kInput×49, fusion:kLoop×26, fusion:kCustom×10
  One residual block, in launch order:
    fusion:kLoop       f32[128]{0}                        loop_rsqrt_fusion.7
    fusion:kInput      f32[128,14,128]{2,1,0}             wrapped_transpose.14
    fusion:kInput      f32[128,14,128]{2,1,0}             wrapped_transpose.15
    fusion:kLoop       f32[128,128,14,128]{3,2,1,0}       loop_maximum_fusion.1
    fusion:kCustom     f32[128,128,3,3]{3,2,1,0}          wrapped_transpose.17
    fusion:kInput      f32[128,256]{1,0}                  input_reduce_fusion.13
    fusion:kInput      f32[128]{0}                        input_reduce_fusion.18
    fusion:kLoop       f32[128]{0}                        loop_multiply_fusion.2
    fusion:kInput      f32[128,256]{1,0}                  input_reduce_fusion.12
    fusion:kInput      f32[128]{0}                        input_reduce_fusion.19
    fusion:kLoop       f32[128]{0}                        loop_rsqrt_fusion.6
    fusion:kInput      f32[128,14,128]{2,1,0}             wrapped_transpose.12
    fusion:kInput      f32[128,14,128]{2,1,0}             wrapped_transpose.13
    fusion:kLoop       f32[128,128,14,128]{3,2,1,0}       loop_maximum_fusion.2
    fusion:kCustom     f32[128,128,3,3]{3,2,1,0}          wrapped_transpose.18
    fusion:kInput      f32[128,256]{1,0}                  input_reduce_fusion.11

## 4. Memory: args 23.1 MB, output 1.84 MB, temp 353.2 MB
  one activation [B,14,128,128] f32 = 117.44 MB

## 5. Where the time goes (ablation inside the full compiled graph, median ms)
  full NRX                               65.318
  without LayerNorm                      58.363
  without the 8 residual convs           12.660
  without both (in/out conv + glue)       3.493
  dispatch floor (jit x+1)                0.139
  ⇒ 8 residual convs ≈ 52.658 ms (81%), 8 LayerNorms ≈ 6.955 ms (11%), dispatch ≈ 0% of a call
