# Compiler walk — backend=tpu (TPU v5 lite), jax 0.12.1, B=128, precision=default

## 1. jaxpr: 186 equations
  broadcast_in_dim×34, add×30, mul×17, reduce_sum×16, div×16, sub×16, conv_general_dilated×10, reshape×10, square×8, sqrt×8, integer_pow×8, custom_jvp_call×8, transpose×1, log×1, real×1, imag×1, concatenate×1

## 2. StableHLO: 321 ops
  broadcast_in_dim×118, constant×50, add×46, divide×24, multiply×17, reduce×16, subtract×16, convolution×10, reshape×10, sqrt×8, transpose×1, log×1, real×1, imag×1, concatenate×1, maximum×1

## 3. Optimized HLO: 39 top-level ops per call (each one a kernel launch on GPU)
  fusion:kLoop×25, fusion:kOutput×9, copy×2, custom-call×2, broadcast×1
  One residual block, in launch order:
    fusion:kLoop       bf16[128]{0:T(256)(128)(2,1)S(1)}  log_multiply_fusion
    broadcast          bf16[128,14,128,1]{0,3,2,1:T(2,128)(2,1)S(1)} broadcast_in_dim.3
    copy               c64[128,2,14,128]{0,1,3,2:T(2,128)S(1)} copy
    custom-call        f32[128,2,14,128]{0,1,3,2:T(2,128)S(1)} custom-call.1
    custom-call        f32[128,2,14,128]{0,1,3,2:T(2,128)S(1)} custom-call
    fusion:kOutput     (f32[128]{0:T(128)S(1)}, f32[128,14,128,128]{0,3,2,1:T(8,128)}) fusion.110
    fusion:kLoop       f32[128]{0:T(128)S(1)}             broadcast_multiply_fusion.7
    fusion:kLoop       f32[128]{0:T(128)S(1)}             multiply_reduce_fusion.7
    fusion:kLoop       f32[128]{0:T(128)S(1)}             add_rsqrt_fusion.7
    fusion:kOutput     (f32[128]{0:T(128)S(1)}, f32[128,14,128,128]{0,3,2,1:T(8,128)}) fusion.111
    fusion:kLoop       f32[128]{0:T(128)S(1)}             broadcast_multiply_fusion.6
    fusion:kLoop       f32[128]{0:T(128)S(1)}             multiply_reduce_fusion.6
    fusion:kLoop       f32[128]{0:T(128)}                 add_rsqrt_fusion.6

## 4. Memory: args 23.1 MB, output 1.84 MB, temp 235.3 MB
  one activation [B,14,128,128] f32 = 117.44 MB

## 5. Where the time goes (ablation inside the full compiled graph, median ms)
  full NRX                                8.902
  without LayerNorm                       5.107
  without the 8 residual convs           11.658
  without both (in/out conv + glue)       0.686
  dispatch floor (jit x+1)                0.191
  ⇒ 8 residual convs ≈ -2.756 ms (-31%), 8 LayerNorms ≈ 3.795 ms (43%), dispatch ≈ 2% of a call
