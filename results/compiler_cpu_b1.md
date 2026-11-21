# Compiler walk — backend=cpu (cpu), jax 0.12.1, B=1, precision=default

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

## 5. Where the time goes (ablation inside the full compiled graph, median ms)
  full NRX                                3.029
  without LayerNorm                       2.483
  without the 8 residual convs            0.812
  without both (in/out conv + glue)       0.283
  dispatch floor (jit x+1)                0.003
  ⇒ 8 residual convs ≈ 2.217 ms (73%), 8 LayerNorms ≈ 0.546 ms (18%), dispatch ≈ 0% of a call
