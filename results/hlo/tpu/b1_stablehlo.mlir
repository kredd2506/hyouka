module @jit_forward attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @empty_mesh = <[]> {stablehlo.mesh = {axes = []}}
  func.func public @main(%arg0: tensor<128xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>}, %arg1: tensor<3x3x5x128xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}, {}]>}, %arg2: tensor<2xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>}, %arg3: tensor<3x3x128x2xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}, {}]>}, %arg4: tensor<128xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>}, %arg5: tensor<3x3x128x128xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}, {}]>}, %arg6: tensor<128xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>}, %arg7: tensor<3x3x128x128xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}, {}]>}, %arg8: tensor<14x128x128xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>}, %arg9: tensor<14x128x128xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>}, %arg10: tensor<14x128x128xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>}, %arg11: tensor<14x128x128xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>}, %arg12: tensor<128xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>}, %arg13: tensor<3x3x128x128xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}, {}]>}, %arg14: tensor<128xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>}, %arg15: tensor<3x3x128x128xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}, {}]>}, %arg16: tensor<14x128x128xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>}, %arg17: tensor<14x128x128xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>}, %arg18: tensor<14x128x128xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>}, %arg19: tensor<14x128x128xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>}, %arg20: tensor<128xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>}, %arg21: tensor<3x3x128x128xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}, {}]>}, %arg22: tensor<128xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>}, %arg23: tensor<3x3x128x128xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}, {}]>}, %arg24: tensor<14x128x128xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>}, %arg25: tensor<14x128x128xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>}, %arg26: tensor<14x128x128xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>}, %arg27: tensor<14x128x128xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>}, %arg28: tensor<128xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>}, %arg29: tensor<3x3x128x128xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}, {}]>}, %arg30: tensor<128xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>}, %arg31: tensor<3x3x128x128xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}, {}]>}, %arg32: tensor<14x128x128xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>}, %arg33: tensor<14x128x128xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>}, %arg34: tensor<14x128x128xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>}, %arg35: tensor<14x128x128xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>}, %arg36: tensor<1x2x14x128xcomplex<f32>> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}, {}]>}, %arg37: tensor<1xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>}) -> (tensor<1x14x128x2xf32> {jax.result_info = "result"}) {
    %0 = stablehlo.transpose %arg36, dims = [0, 2, 3, 1] : (tensor<1x2x14x128xcomplex<f32>>) -> tensor<1x14x128x2xcomplex<f32>>
    %1 = stablehlo.log %arg37 : tensor<1xf32>
    %cst = stablehlo.constant dense<0.434294492> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<1xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<1xf32>) -> tensor<1x1x1x1xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1, 2, 3] : (tensor<1x1x1x1xf32>) -> tensor<1x14x128x1xf32>
    %6 = stablehlo.real %0 : (tensor<1x14x128x2xcomplex<f32>>) -> tensor<1x14x128x2xf32>
    %7 = stablehlo.imag %0 : (tensor<1x14x128x2xcomplex<f32>>) -> tensor<1x14x128x2xf32>
    %8 = stablehlo.concatenate %6, %7, %5, dim = 3 : (tensor<1x14x128x2xf32>, tensor<1x14x128x2xf32>, tensor<1x14x128x1xf32>) -> tensor<1x14x128x5xf32>
    %9 = stablehlo.convolution(%8, %arg1) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x14x128x5xf32>, tensor<3x3x5x128xf32>) -> tensor<1x14x128x128xf32>
    %10 = stablehlo.reshape %arg0 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<1x14x128x128xf32>
    %12 = stablehlo.add %9, %11 : tensor<1x14x128x128xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %13 = stablehlo.reduce(%12 init: %cst_0) applies stablehlo.add across dimensions = [1, 2, 3] : (tensor<1x14x128x128xf32>, tensor<f32>) -> tensor<1xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0] : (tensor<1xf32>) -> tensor<1x1x1x1xf32>
    %cst_1 = stablehlo.constant dense<2.293760e+05> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x1x1x1xf32>
    %16 = stablehlo.divide %14, %15 : tensor<1x1x1x1xf32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [0, 1, 2, 3] : (tensor<1x1x1x1xf32>) -> tensor<1x14x128x128xf32>
    %18 = stablehlo.subtract %12, %17 : tensor<1x14x128x128xf32>
    %19 = chlo.square %18 : tensor<1x14x128x128xf32> -> tensor<1x14x128x128xf32>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %20 = stablehlo.reduce(%19 init: %cst_2) applies stablehlo.add across dimensions = [1, 2, 3] : (tensor<1x14x128x128xf32>, tensor<f32>) -> tensor<1xf32>
    %21 = stablehlo.broadcast_in_dim %20, dims = [0] : (tensor<1xf32>) -> tensor<1x1x1x1xf32>
    %cst_3 = stablehlo.constant dense<2.293760e+05> : tensor<f32>
    %22 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x1x1x1xf32>
    %23 = stablehlo.divide %21, %22 : tensor<1x1x1x1xf32>
    %24 = stablehlo.broadcast_in_dim %16, dims = [0, 1, 2, 3] : (tensor<1x1x1x1xf32>) -> tensor<1x14x128x128xf32>
    %25 = stablehlo.subtract %12, %24 : tensor<1x14x128x128xf32>
    %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %26 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x1x1x1xf32>
    %27 = stablehlo.add %23, %26 : tensor<1x1x1x1xf32>
    %28 = stablehlo.sqrt %27 : tensor<1x1x1x1xf32>
    %cst_5 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %29 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x1x1x1xf32>
    %30 = stablehlo.divide %29, %28 : tensor<1x1x1x1xf32>
    %31 = stablehlo.broadcast_in_dim %30, dims = [0, 1, 2, 3] : (tensor<1x1x1x1xf32>) -> tensor<1x14x128x128xf32>
    %32 = stablehlo.multiply %25, %31 : tensor<1x14x128x128xf32>
    %33 = stablehlo.broadcast_in_dim %arg9, dims = [1, 2, 3] : (tensor<14x128x128xf32>) -> tensor<1x14x128x128xf32>
    %34 = stablehlo.multiply %32, %33 : tensor<1x14x128x128xf32>
    %35 = stablehlo.broadcast_in_dim %arg8, dims = [1, 2, 3] : (tensor<14x128x128xf32>) -> tensor<1x14x128x128xf32>
    %36 = stablehlo.add %34, %35 : tensor<1x14x128x128xf32>
    %37 = call @relu(%36) : (tensor<1x14x128x128xf32>) -> tensor<1x14x128x128xf32>
    %38 = stablehlo.convolution(%37, %arg5) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x14x128x128xf32>, tensor<3x3x128x128xf32>) -> tensor<1x14x128x128xf32>
    %39 = stablehlo.reshape %arg4 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %40 = stablehlo.broadcast_in_dim %39, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<1x14x128x128xf32>
    %41 = stablehlo.add %38, %40 : tensor<1x14x128x128xf32>
    %cst_6 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %42 = stablehlo.reduce(%41 init: %cst_6) applies stablehlo.add across dimensions = [1, 2, 3] : (tensor<1x14x128x128xf32>, tensor<f32>) -> tensor<1xf32>
    %43 = stablehlo.broadcast_in_dim %42, dims = [0] : (tensor<1xf32>) -> tensor<1x1x1x1xf32>
    %cst_7 = stablehlo.constant dense<2.293760e+05> : tensor<f32>
    %44 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1x1x1x1xf32>
    %45 = stablehlo.divide %43, %44 : tensor<1x1x1x1xf32>
    %46 = stablehlo.broadcast_in_dim %45, dims = [0, 1, 2, 3] : (tensor<1x1x1x1xf32>) -> tensor<1x14x128x128xf32>
    %47 = stablehlo.subtract %41, %46 : tensor<1x14x128x128xf32>
    %48 = chlo.square %47 : tensor<1x14x128x128xf32> -> tensor<1x14x128x128xf32>
    %cst_8 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %49 = stablehlo.reduce(%48 init: %cst_8) applies stablehlo.add across dimensions = [1, 2, 3] : (tensor<1x14x128x128xf32>, tensor<f32>) -> tensor<1xf32>
    %50 = stablehlo.broadcast_in_dim %49, dims = [0] : (tensor<1xf32>) -> tensor<1x1x1x1xf32>
    %cst_9 = stablehlo.constant dense<2.293760e+05> : tensor<f32>
    %51 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x1x1x1xf32>
    %52 = stablehlo.divide %50, %51 : tensor<1x1x1x1xf32>
    %53 = stablehlo.broadcast_in_dim %45, dims = [0, 1, 2, 3] : (tensor<1x1x1x1xf32>) -> tensor<1x14x128x128xf32>
    %54 = stablehlo.subtract %41, %53 : tensor<1x14x128x128xf32>
    %cst_10 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %55 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x1x1x1xf32>
    %56 = stablehlo.add %52, %55 : tensor<1x1x1x1xf32>
    %57 = stablehlo.sqrt %56 : tensor<1x1x1x1xf32>
    %cst_11 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %58 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x1x1x1xf32>
    %59 = stablehlo.divide %58, %57 : tensor<1x1x1x1xf32>
    %60 = stablehlo.broadcast_in_dim %59, dims = [0, 1, 2, 3] : (tensor<1x1x1x1xf32>) -> tensor<1x14x128x128xf32>
    %61 = stablehlo.multiply %54, %60 : tensor<1x14x128x128xf32>
    %62 = stablehlo.broadcast_in_dim %arg11, dims = [1, 2, 3] : (tensor<14x128x128xf32>) -> tensor<1x14x128x128xf32>
    %63 = stablehlo.multiply %61, %62 : tensor<1x14x128x128xf32>
    %64 = stablehlo.broadcast_in_dim %arg10, dims = [1, 2, 3] : (tensor<14x128x128xf32>) -> tensor<1x14x128x128xf32>
    %65 = stablehlo.add %63, %64 : tensor<1x14x128x128xf32>
    %66 = call @relu(%65) : (tensor<1x14x128x128xf32>) -> tensor<1x14x128x128xf32>
    %67 = stablehlo.convolution(%66, %arg7) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x14x128x128xf32>, tensor<3x3x128x128xf32>) -> tensor<1x14x128x128xf32>
    %68 = stablehlo.reshape %arg6 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %69 = stablehlo.broadcast_in_dim %68, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<1x14x128x128xf32>
    %70 = stablehlo.add %67, %69 : tensor<1x14x128x128xf32>
    %71 = stablehlo.add %70, %12 : tensor<1x14x128x128xf32>
    %cst_12 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %72 = stablehlo.reduce(%71 init: %cst_12) applies stablehlo.add across dimensions = [1, 2, 3] : (tensor<1x14x128x128xf32>, tensor<f32>) -> tensor<1xf32>
    %73 = stablehlo.broadcast_in_dim %72, dims = [0] : (tensor<1xf32>) -> tensor<1x1x1x1xf32>
    %cst_13 = stablehlo.constant dense<2.293760e+05> : tensor<f32>
    %74 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x1x1x1xf32>
    %75 = stablehlo.divide %73, %74 : tensor<1x1x1x1xf32>
    %76 = stablehlo.broadcast_in_dim %75, dims = [0, 1, 2, 3] : (tensor<1x1x1x1xf32>) -> tensor<1x14x128x128xf32>
    %77 = stablehlo.subtract %71, %76 : tensor<1x14x128x128xf32>
    %78 = chlo.square %77 : tensor<1x14x128x128xf32> -> tensor<1x14x128x128xf32>
    %cst_14 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %79 = stablehlo.reduce(%78 init: %cst_14) applies stablehlo.add across dimensions = [1, 2, 3] : (tensor<1x14x128x128xf32>, tensor<f32>) -> tensor<1xf32>
    %80 = stablehlo.broadcast_in_dim %79, dims = [0] : (tensor<1xf32>) -> tensor<1x1x1x1xf32>
    %cst_15 = stablehlo.constant dense<2.293760e+05> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<1x1x1x1xf32>
    %82 = stablehlo.divide %80, %81 : tensor<1x1x1x1xf32>
    %83 = stablehlo.broadcast_in_dim %75, dims = [0, 1, 2, 3] : (tensor<1x1x1x1xf32>) -> tensor<1x14x128x128xf32>
    %84 = stablehlo.subtract %71, %83 : tensor<1x14x128x128xf32>
    %cst_16 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %85 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<1x1x1x1xf32>
    %86 = stablehlo.add %82, %85 : tensor<1x1x1x1xf32>
    %87 = stablehlo.sqrt %86 : tensor<1x1x1x1xf32>
    %cst_17 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %88 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<1x1x1x1xf32>
    %89 = stablehlo.divide %88, %87 : tensor<1x1x1x1xf32>
    %90 = stablehlo.broadcast_in_dim %89, dims = [0, 1, 2, 3] : (tensor<1x1x1x1xf32>) -> tensor<1x14x128x128xf32>
    %91 = stablehlo.multiply %84, %90 : tensor<1x14x128x128xf32>
    %92 = stablehlo.broadcast_in_dim %arg17, dims = [1, 2, 3] : (tensor<14x128x128xf32>) -> tensor<1x14x128x128xf32>
    %93 = stablehlo.multiply %91, %92 : tensor<1x14x128x128xf32>
    %94 = stablehlo.broadcast_in_dim %arg16, dims = [1, 2, 3] : (tensor<14x128x128xf32>) -> tensor<1x14x128x128xf32>
    %95 = stablehlo.add %93, %94 : tensor<1x14x128x128xf32>
    %96 = call @relu(%95) : (tensor<1x14x128x128xf32>) -> tensor<1x14x128x128xf32>
    %97 = stablehlo.convolution(%96, %arg13) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x14x128x128xf32>, tensor<3x3x128x128xf32>) -> tensor<1x14x128x128xf32>
    %98 = stablehlo.reshape %arg12 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %99 = stablehlo.broadcast_in_dim %98, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<1x14x128x128xf32>
    %100 = stablehlo.add %97, %99 : tensor<1x14x128x128xf32>
    %cst_18 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %101 = stablehlo.reduce(%100 init: %cst_18) applies stablehlo.add across dimensions = [1, 2, 3] : (tensor<1x14x128x128xf32>, tensor<f32>) -> tensor<1xf32>
    %102 = stablehlo.broadcast_in_dim %101, dims = [0] : (tensor<1xf32>) -> tensor<1x1x1x1xf32>
    %cst_19 = stablehlo.constant dense<2.293760e+05> : tensor<f32>
    %103 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<1x1x1x1xf32>
    %104 = stablehlo.divide %102, %103 : tensor<1x1x1x1xf32>
    %105 = stablehlo.broadcast_in_dim %104, dims = [0, 1, 2, 3] : (tensor<1x1x1x1xf32>) -> tensor<1x14x128x128xf32>
    %106 = stablehlo.subtract %100, %105 : tensor<1x14x128x128xf32>
    %107 = chlo.square %106 : tensor<1x14x128x128xf32> -> tensor<1x14x128x128xf32>
    %cst_20 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %108 = stablehlo.reduce(%107 init: %cst_20) applies stablehlo.add across dimensions = [1, 2, 3] : (tensor<1x14x128x128xf32>, tensor<f32>) -> tensor<1xf32>
    %109 = stablehlo.broadcast_in_dim %108, dims = [0] : (tensor<1xf32>) -> tensor<1x1x1x1xf32>
    %cst_21 = stablehlo.constant dense<2.293760e+05> : tensor<f32>
    %110 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<1x1x1x1xf32>
    %111 = stablehlo.divide %109, %110 : tensor<1x1x1x1xf32>
    %112 = stablehlo.broadcast_in_dim %104, dims = [0, 1, 2, 3] : (tensor<1x1x1x1xf32>) -> tensor<1x14x128x128xf32>
    %113 = stablehlo.subtract %100, %112 : tensor<1x14x128x128xf32>
    %cst_22 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %114 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<1x1x1x1xf32>
    %115 = stablehlo.add %111, %114 : tensor<1x1x1x1xf32>
    %116 = stablehlo.sqrt %115 : tensor<1x1x1x1xf32>
    %cst_23 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %117 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<1x1x1x1xf32>
    %118 = stablehlo.divide %117, %116 : tensor<1x1x1x1xf32>
    %119 = stablehlo.broadcast_in_dim %118, dims = [0, 1, 2, 3] : (tensor<1x1x1x1xf32>) -> tensor<1x14x128x128xf32>
    %120 = stablehlo.multiply %113, %119 : tensor<1x14x128x128xf32>
    %121 = stablehlo.broadcast_in_dim %arg19, dims = [1, 2, 3] : (tensor<14x128x128xf32>) -> tensor<1x14x128x128xf32>
    %122 = stablehlo.multiply %120, %121 : tensor<1x14x128x128xf32>
    %123 = stablehlo.broadcast_in_dim %arg18, dims = [1, 2, 3] : (tensor<14x128x128xf32>) -> tensor<1x14x128x128xf32>
    %124 = stablehlo.add %122, %123 : tensor<1x14x128x128xf32>
    %125 = call @relu(%124) : (tensor<1x14x128x128xf32>) -> tensor<1x14x128x128xf32>
    %126 = stablehlo.convolution(%125, %arg15) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x14x128x128xf32>, tensor<3x3x128x128xf32>) -> tensor<1x14x128x128xf32>
    %127 = stablehlo.reshape %arg14 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %128 = stablehlo.broadcast_in_dim %127, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<1x14x128x128xf32>
    %129 = stablehlo.add %126, %128 : tensor<1x14x128x128xf32>
    %130 = stablehlo.add %129, %71 : tensor<1x14x128x128xf32>
    %cst_24 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %131 = stablehlo.reduce(%130 init: %cst_24) applies stablehlo.add across dimensions = [1, 2, 3] : (tensor<1x14x128x128xf32>, tensor<f32>) -> tensor<1xf32>
    %132 = stablehlo.broadcast_in_dim %131, dims = [0] : (tensor<1xf32>) -> tensor<1x1x1x1xf32>
    %cst_25 = stablehlo.constant dense<2.293760e+05> : tensor<f32>
    %133 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<1x1x1x1xf32>
    %134 = stablehlo.divide %132, %133 : tensor<1x1x1x1xf32>
    %135 = stablehlo.broadcast_in_dim %134, dims = [0, 1, 2, 3] : (tensor<1x1x1x1xf32>) -> tensor<1x14x128x128xf32>
    %136 = stablehlo.subtract %130, %135 : tensor<1x14x128x128xf32>
    %137 = chlo.square %136 : tensor<1x14x128x128xf32> -> tensor<1x14x128x128xf32>
    %cst_26 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %138 = stablehlo.reduce(%137 init: %cst_26) applies stablehlo.add across dimensions = [1, 2, 3] : (tensor<1x14x128x128xf32>, tensor<f32>) -> tensor<1xf32>
    %139 = stablehlo.broadcast_in_dim %138, dims = [0] : (tensor<1xf32>) -> tensor<1x1x1x1xf32>
    %cst_27 = stablehlo.constant dense<2.293760e+05> : tensor<f32>
    %140 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x1x1x1xf32>
    %141 = stablehlo.divide %139, %140 : tensor<1x1x1x1xf32>
    %142 = stablehlo.broadcast_in_dim %134, dims = [0, 1, 2, 3] : (tensor<1x1x1x1xf32>) -> tensor<1x14x128x128xf32>
    %143 = stablehlo.subtract %130, %142 : tensor<1x14x128x128xf32>
    %cst_28 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %144 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x1x1x1xf32>
    %145 = stablehlo.add %141, %144 : tensor<1x1x1x1xf32>
    %146 = stablehlo.sqrt %145 : tensor<1x1x1x1xf32>
    %cst_29 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %147 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x1x1x1xf32>
    %148 = stablehlo.divide %147, %146 : tensor<1x1x1x1xf32>
    %149 = stablehlo.broadcast_in_dim %148, dims = [0, 1, 2, 3] : (tensor<1x1x1x1xf32>) -> tensor<1x14x128x128xf32>
    %150 = stablehlo.multiply %143, %149 : tensor<1x14x128x128xf32>
    %151 = stablehlo.broadcast_in_dim %arg25, dims = [1, 2, 3] : (tensor<14x128x128xf32>) -> tensor<1x14x128x128xf32>
    %152 = stablehlo.multiply %150, %151 : tensor<1x14x128x128xf32>
    %153 = stablehlo.broadcast_in_dim %arg24, dims = [1, 2, 3] : (tensor<14x128x128xf32>) -> tensor<1x14x128x128xf32>
    %154 = stablehlo.add %152, %153 : tensor<1x14x128x128xf32>
    %155 = call @relu(%154) : (tensor<1x14x128x128xf32>) -> tensor<1x14x128x128xf32>
    %156 = stablehlo.convolution(%155, %arg21) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x14x128x128xf32>, tensor<3x3x128x128xf32>) -> tensor<1x14x128x128xf32>
    %157 = stablehlo.reshape %arg20 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %158 = stablehlo.broadcast_in_dim %157, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<1x14x128x128xf32>
    %159 = stablehlo.add %156, %158 : tensor<1x14x128x128xf32>
    %cst_30 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %160 = stablehlo.reduce(%159 init: %cst_30) applies stablehlo.add across dimensions = [1, 2, 3] : (tensor<1x14x128x128xf32>, tensor<f32>) -> tensor<1xf32>
    %161 = stablehlo.broadcast_in_dim %160, dims = [0] : (tensor<1xf32>) -> tensor<1x1x1x1xf32>
    %cst_31 = stablehlo.constant dense<2.293760e+05> : tensor<f32>
    %162 = stablehlo.broadcast_in_dim %cst_31, dims = [] : (tensor<f32>) -> tensor<1x1x1x1xf32>
    %163 = stablehlo.divide %161, %162 : tensor<1x1x1x1xf32>
    %164 = stablehlo.broadcast_in_dim %163, dims = [0, 1, 2, 3] : (tensor<1x1x1x1xf32>) -> tensor<1x14x128x128xf32>
    %165 = stablehlo.subtract %159, %164 : tensor<1x14x128x128xf32>
    %166 = chlo.square %165 : tensor<1x14x128x128xf32> -> tensor<1x14x128x128xf32>
    %cst_32 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %167 = stablehlo.reduce(%166 init: %cst_32) applies stablehlo.add across dimensions = [1, 2, 3] : (tensor<1x14x128x128xf32>, tensor<f32>) -> tensor<1xf32>
    %168 = stablehlo.broadcast_in_dim %167, dims = [0] : (tensor<1xf32>) -> tensor<1x1x1x1xf32>
    %cst_33 = stablehlo.constant dense<2.293760e+05> : tensor<f32>
    %169 = stablehlo.broadcast_in_dim %cst_33, dims = [] : (tensor<f32>) -> tensor<1x1x1x1xf32>
    %170 = stablehlo.divide %168, %169 : tensor<1x1x1x1xf32>
    %171 = stablehlo.broadcast_in_dim %163, dims = [0, 1, 2, 3] : (tensor<1x1x1x1xf32>) -> tensor<1x14x128x128xf32>
    %172 = stablehlo.subtract %159, %171 : tensor<1x14x128x128xf32>
    %cst_34 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %173 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<1x1x1x1xf32>
    %174 = stablehlo.add %170, %173 : tensor<1x1x1x1xf32>
    %175 = stablehlo.sqrt %174 : tensor<1x1x1x1xf32>
    %cst_35 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %176 = stablehlo.broadcast_in_dim %cst_35, dims = [] : (tensor<f32>) -> tensor<1x1x1x1xf32>
    %177 = stablehlo.divide %176, %175 : tensor<1x1x1x1xf32>
    %178 = stablehlo.broadcast_in_dim %177, dims = [0, 1, 2, 3] : (tensor<1x1x1x1xf32>) -> tensor<1x14x128x128xf32>
    %179 = stablehlo.multiply %172, %178 : tensor<1x14x128x128xf32>
    %180 = stablehlo.broadcast_in_dim %arg27, dims = [1, 2, 3] : (tensor<14x128x128xf32>) -> tensor<1x14x128x128xf32>
    %181 = stablehlo.multiply %179, %180 : tensor<1x14x128x128xf32>
    %182 = stablehlo.broadcast_in_dim %arg26, dims = [1, 2, 3] : (tensor<14x128x128xf32>) -> tensor<1x14x128x128xf32>
    %183 = stablehlo.add %181, %182 : tensor<1x14x128x128xf32>
    %184 = call @relu(%183) : (tensor<1x14x128x128xf32>) -> tensor<1x14x128x128xf32>
    %185 = stablehlo.convolution(%184, %arg23) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x14x128x128xf32>, tensor<3x3x128x128xf32>) -> tensor<1x14x128x128xf32>
    %186 = stablehlo.reshape %arg22 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %187 = stablehlo.broadcast_in_dim %186, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<1x14x128x128xf32>
    %188 = stablehlo.add %185, %187 : tensor<1x14x128x128xf32>
    %189 = stablehlo.add %188, %130 : tensor<1x14x128x128xf32>
    %cst_36 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %190 = stablehlo.reduce(%189 init: %cst_36) applies stablehlo.add across dimensions = [1, 2, 3] : (tensor<1x14x128x128xf32>, tensor<f32>) -> tensor<1xf32>
    %191 = stablehlo.broadcast_in_dim %190, dims = [0] : (tensor<1xf32>) -> tensor<1x1x1x1xf32>
    %cst_37 = stablehlo.constant dense<2.293760e+05> : tensor<f32>
    %192 = stablehlo.broadcast_in_dim %cst_37, dims = [] : (tensor<f32>) -> tensor<1x1x1x1xf32>
    %193 = stablehlo.divide %191, %192 : tensor<1x1x1x1xf32>
    %194 = stablehlo.broadcast_in_dim %193, dims = [0, 1, 2, 3] : (tensor<1x1x1x1xf32>) -> tensor<1x14x128x128xf32>
    %195 = stablehlo.subtract %189, %194 : tensor<1x14x128x128xf32>
    %196 = chlo.square %195 : tensor<1x14x128x128xf32> -> tensor<1x14x128x128xf32>
    %cst_38 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %197 = stablehlo.reduce(%196 init: %cst_38) applies stablehlo.add across dimensions = [1, 2, 3] : (tensor<1x14x128x128xf32>, tensor<f32>) -> tensor<1xf32>
    %198 = stablehlo.broadcast_in_dim %197, dims = [0] : (tensor<1xf32>) -> tensor<1x1x1x1xf32>
    %cst_39 = stablehlo.constant dense<2.293760e+05> : tensor<f32>
    %199 = stablehlo.broadcast_in_dim %cst_39, dims = [] : (tensor<f32>) -> tensor<1x1x1x1xf32>
    %200 = stablehlo.divide %198, %199 : tensor<1x1x1x1xf32>
    %201 = stablehlo.broadcast_in_dim %193, dims = [0, 1, 2, 3] : (tensor<1x1x1x1xf32>) -> tensor<1x14x128x128xf32>
    %202 = stablehlo.subtract %189, %201 : tensor<1x14x128x128xf32>
    %cst_40 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %203 = stablehlo.broadcast_in_dim %cst_40, dims = [] : (tensor<f32>) -> tensor<1x1x1x1xf32>
    %204 = stablehlo.add %200, %203 : tensor<1x1x1x1xf32>
    %205 = stablehlo.sqrt %204 : tensor<1x1x1x1xf32>
    %cst_41 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %206 = stablehlo.broadcast_in_dim %cst_41, dims = [] : (tensor<f32>) -> tensor<1x1x1x1xf32>
    %207 = stablehlo.divide %206, %205 : tensor<1x1x1x1xf32>
    %208 = stablehlo.broadcast_in_dim %207, dims = [0, 1, 2, 3] : (tensor<1x1x1x1xf32>) -> tensor<1x14x128x128xf32>
    %209 = stablehlo.multiply %202, %208 : tensor<1x14x128x128xf32>
    %210 = stablehlo.broadcast_in_dim %arg33, dims = [1, 2, 3] : (tensor<14x128x128xf32>) -> tensor<1x14x128x128xf32>
    %211 = stablehlo.multiply %209, %210 : tensor<1x14x128x128xf32>
    %212 = stablehlo.broadcast_in_dim %arg32, dims = [1, 2, 3] : (tensor<14x128x128xf32>) -> tensor<1x14x128x128xf32>
    %213 = stablehlo.add %211, %212 : tensor<1x14x128x128xf32>
    %214 = call @relu(%213) : (tensor<1x14x128x128xf32>) -> tensor<1x14x128x128xf32>
    %215 = stablehlo.convolution(%214, %arg29) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x14x128x128xf32>, tensor<3x3x128x128xf32>) -> tensor<1x14x128x128xf32>
    %216 = stablehlo.reshape %arg28 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %217 = stablehlo.broadcast_in_dim %216, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<1x14x128x128xf32>
    %218 = stablehlo.add %215, %217 : tensor<1x14x128x128xf32>
    %cst_42 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %219 = stablehlo.reduce(%218 init: %cst_42) applies stablehlo.add across dimensions = [1, 2, 3] : (tensor<1x14x128x128xf32>, tensor<f32>) -> tensor<1xf32>
    %220 = stablehlo.broadcast_in_dim %219, dims = [0] : (tensor<1xf32>) -> tensor<1x1x1x1xf32>
    %cst_43 = stablehlo.constant dense<2.293760e+05> : tensor<f32>
    %221 = stablehlo.broadcast_in_dim %cst_43, dims = [] : (tensor<f32>) -> tensor<1x1x1x1xf32>
    %222 = stablehlo.divide %220, %221 : tensor<1x1x1x1xf32>
    %223 = stablehlo.broadcast_in_dim %222, dims = [0, 1, 2, 3] : (tensor<1x1x1x1xf32>) -> tensor<1x14x128x128xf32>
    %224 = stablehlo.subtract %218, %223 : tensor<1x14x128x128xf32>
    %225 = chlo.square %224 : tensor<1x14x128x128xf32> -> tensor<1x14x128x128xf32>
    %cst_44 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %226 = stablehlo.reduce(%225 init: %cst_44) applies stablehlo.add across dimensions = [1, 2, 3] : (tensor<1x14x128x128xf32>, tensor<f32>) -> tensor<1xf32>
    %227 = stablehlo.broadcast_in_dim %226, dims = [0] : (tensor<1xf32>) -> tensor<1x1x1x1xf32>
    %cst_45 = stablehlo.constant dense<2.293760e+05> : tensor<f32>
    %228 = stablehlo.broadcast_in_dim %cst_45, dims = [] : (tensor<f32>) -> tensor<1x1x1x1xf32>
    %229 = stablehlo.divide %227, %228 : tensor<1x1x1x1xf32>
    %230 = stablehlo.broadcast_in_dim %222, dims = [0, 1, 2, 3] : (tensor<1x1x1x1xf32>) -> tensor<1x14x128x128xf32>
    %231 = stablehlo.subtract %218, %230 : tensor<1x14x128x128xf32>
    %cst_46 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %232 = stablehlo.broadcast_in_dim %cst_46, dims = [] : (tensor<f32>) -> tensor<1x1x1x1xf32>
    %233 = stablehlo.add %229, %232 : tensor<1x1x1x1xf32>
    %234 = stablehlo.sqrt %233 : tensor<1x1x1x1xf32>
    %cst_47 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %235 = stablehlo.broadcast_in_dim %cst_47, dims = [] : (tensor<f32>) -> tensor<1x1x1x1xf32>
    %236 = stablehlo.divide %235, %234 : tensor<1x1x1x1xf32>
    %237 = stablehlo.broadcast_in_dim %236, dims = [0, 1, 2, 3] : (tensor<1x1x1x1xf32>) -> tensor<1x14x128x128xf32>
    %238 = stablehlo.multiply %231, %237 : tensor<1x14x128x128xf32>
    %239 = stablehlo.broadcast_in_dim %arg35, dims = [1, 2, 3] : (tensor<14x128x128xf32>) -> tensor<1x14x128x128xf32>
    %240 = stablehlo.multiply %238, %239 : tensor<1x14x128x128xf32>
    %241 = stablehlo.broadcast_in_dim %arg34, dims = [1, 2, 3] : (tensor<14x128x128xf32>) -> tensor<1x14x128x128xf32>
    %242 = stablehlo.add %240, %241 : tensor<1x14x128x128xf32>
    %243 = call @relu(%242) : (tensor<1x14x128x128xf32>) -> tensor<1x14x128x128xf32>
    %244 = stablehlo.convolution(%243, %arg31) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x14x128x128xf32>, tensor<3x3x128x128xf32>) -> tensor<1x14x128x128xf32>
    %245 = stablehlo.reshape %arg30 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %246 = stablehlo.broadcast_in_dim %245, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<1x14x128x128xf32>
    %247 = stablehlo.add %244, %246 : tensor<1x14x128x128xf32>
    %248 = stablehlo.add %247, %189 : tensor<1x14x128x128xf32>
    %249 = stablehlo.convolution(%248, %arg3) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x14x128x128xf32>, tensor<3x3x128x2xf32>) -> tensor<1x14x128x2xf32>
    %250 = stablehlo.reshape %arg2 : (tensor<2xf32>) -> tensor<1x1x1x2xf32>
    %251 = stablehlo.broadcast_in_dim %250, dims = [0, 1, 2, 3] : (tensor<1x1x1x2xf32>) -> tensor<1x14x128x2xf32>
    %252 = stablehlo.add %249, %251 : tensor<1x14x128x2xf32>
    return %252 : tensor<1x14x128x2xf32>
  }
  func.func private @relu(%arg0: tensor<1x14x128x128xf32>) -> tensor<1x14x128x128xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x14x128x128xf32>
    %1 = stablehlo.maximum %arg0, %0 : tensor<1x14x128x128xf32>
    return %1 : tensor<1x14x128x128xf32>
  }
}
