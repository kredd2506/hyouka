module @jit_forward attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @empty_mesh = <[]> {stablehlo.mesh = {axes = []}}
  func.func public @main(%arg0: tensor<128xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>}, %arg1: tensor<3x3x5x128xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}, {}]>}, %arg2: tensor<2xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>}, %arg3: tensor<3x3x128x2xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}, {}]>}, %arg4: tensor<128xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>}, %arg5: tensor<3x3x128x128xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}, {}]>}, %arg6: tensor<128xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>}, %arg7: tensor<3x3x128x128xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}, {}]>}, %arg8: tensor<14x128x128xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>}, %arg9: tensor<14x128x128xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>}, %arg10: tensor<14x128x128xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>}, %arg11: tensor<14x128x128xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>}, %arg12: tensor<128xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>}, %arg13: tensor<3x3x128x128xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}, {}]>}, %arg14: tensor<128xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>}, %arg15: tensor<3x3x128x128xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}, {}]>}, %arg16: tensor<14x128x128xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>}, %arg17: tensor<14x128x128xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>}, %arg18: tensor<14x128x128xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>}, %arg19: tensor<14x128x128xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>}, %arg20: tensor<128xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>}, %arg21: tensor<3x3x128x128xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}, {}]>}, %arg22: tensor<128xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>}, %arg23: tensor<3x3x128x128xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}, {}]>}, %arg24: tensor<14x128x128xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>}, %arg25: tensor<14x128x128xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>}, %arg26: tensor<14x128x128xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>}, %arg27: tensor<14x128x128xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>}, %arg28: tensor<128xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>}, %arg29: tensor<3x3x128x128xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}, {}]>}, %arg30: tensor<128xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>}, %arg31: tensor<3x3x128x128xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}, {}]>}, %arg32: tensor<14x128x128xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>}, %arg33: tensor<14x128x128xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>}, %arg34: tensor<14x128x128xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>}, %arg35: tensor<14x128x128xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}]>}, %arg36: tensor<8x2x14x128xcomplex<f32>> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}, {}, {}]>}, %arg37: tensor<8xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>}) -> (tensor<8x14x128x2xf32> {jax.result_info = "result"}) {
    %0 = stablehlo.transpose %arg36, dims = [0, 2, 3, 1] : (tensor<8x2x14x128xcomplex<f32>>) -> tensor<8x14x128x2xcomplex<f32>>
    %1 = stablehlo.log %arg37 : tensor<8xf32>
    %cst = stablehlo.constant dense<0.434294492> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<8xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<8xf32>) -> tensor<8x1x1x1xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1, 2, 3] : (tensor<8x1x1x1xf32>) -> tensor<8x14x128x1xf32>
    %6 = stablehlo.real %0 : (tensor<8x14x128x2xcomplex<f32>>) -> tensor<8x14x128x2xf32>
    %7 = stablehlo.imag %0 : (tensor<8x14x128x2xcomplex<f32>>) -> tensor<8x14x128x2xf32>
    %8 = stablehlo.concatenate %6, %7, %5, dim = 3 : (tensor<8x14x128x2xf32>, tensor<8x14x128x2xf32>, tensor<8x14x128x1xf32>) -> tensor<8x14x128x5xf32>
    %9 = stablehlo.convolution(%8, %arg1) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision HIGHEST>, #stablehlo<precision HIGHEST>]} : (tensor<8x14x128x5xf32>, tensor<3x3x5x128xf32>) -> tensor<8x14x128x128xf32>
    %10 = stablehlo.reshape %arg0 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<8x14x128x128xf32>
    %12 = stablehlo.add %9, %11 : tensor<8x14x128x128xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %13 = stablehlo.reduce(%12 init: %cst_0) applies stablehlo.add across dimensions = [1, 2, 3] : (tensor<8x14x128x128xf32>, tensor<f32>) -> tensor<8xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0] : (tensor<8xf32>) -> tensor<8x1x1x1xf32>
    %cst_1 = stablehlo.constant dense<2.293760e+05> : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<8x1x1x1xf32>
    %16 = stablehlo.divide %14, %15 : tensor<8x1x1x1xf32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [0, 1, 2, 3] : (tensor<8x1x1x1xf32>) -> tensor<8x14x128x128xf32>
    %18 = stablehlo.subtract %12, %17 : tensor<8x14x128x128xf32>
    %19 = chlo.square %18 : tensor<8x14x128x128xf32> -> tensor<8x14x128x128xf32>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %20 = stablehlo.reduce(%19 init: %cst_2) applies stablehlo.add across dimensions = [1, 2, 3] : (tensor<8x14x128x128xf32>, tensor<f32>) -> tensor<8xf32>
    %21 = stablehlo.broadcast_in_dim %20, dims = [0] : (tensor<8xf32>) -> tensor<8x1x1x1xf32>
    %cst_3 = stablehlo.constant dense<2.293760e+05> : tensor<f32>
    %22 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x1x1x1xf32>
    %23 = stablehlo.divide %21, %22 : tensor<8x1x1x1xf32>
    %24 = stablehlo.broadcast_in_dim %16, dims = [0, 1, 2, 3] : (tensor<8x1x1x1xf32>) -> tensor<8x14x128x128xf32>
    %25 = stablehlo.subtract %12, %24 : tensor<8x14x128x128xf32>
    %cst_4 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %26 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<8x1x1x1xf32>
    %27 = stablehlo.add %23, %26 : tensor<8x1x1x1xf32>
    %28 = stablehlo.sqrt %27 : tensor<8x1x1x1xf32>
    %cst_5 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %29 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<8x1x1x1xf32>
    %30 = stablehlo.divide %29, %28 : tensor<8x1x1x1xf32>
    %31 = stablehlo.broadcast_in_dim %30, dims = [0, 1, 2, 3] : (tensor<8x1x1x1xf32>) -> tensor<8x14x128x128xf32>
    %32 = stablehlo.multiply %25, %31 : tensor<8x14x128x128xf32>
    %33 = stablehlo.broadcast_in_dim %arg9, dims = [1, 2, 3] : (tensor<14x128x128xf32>) -> tensor<1x14x128x128xf32>
    %34 = stablehlo.broadcast_in_dim %33, dims = [0, 1, 2, 3] : (tensor<1x14x128x128xf32>) -> tensor<8x14x128x128xf32>
    %35 = stablehlo.multiply %32, %34 : tensor<8x14x128x128xf32>
    %36 = stablehlo.broadcast_in_dim %arg8, dims = [1, 2, 3] : (tensor<14x128x128xf32>) -> tensor<1x14x128x128xf32>
    %37 = stablehlo.broadcast_in_dim %36, dims = [0, 1, 2, 3] : (tensor<1x14x128x128xf32>) -> tensor<8x14x128x128xf32>
    %38 = stablehlo.add %35, %37 : tensor<8x14x128x128xf32>
    %39 = call @relu(%38) : (tensor<8x14x128x128xf32>) -> tensor<8x14x128x128xf32>
    %40 = stablehlo.convolution(%39, %arg5) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision HIGHEST>, #stablehlo<precision HIGHEST>]} : (tensor<8x14x128x128xf32>, tensor<3x3x128x128xf32>) -> tensor<8x14x128x128xf32>
    %41 = stablehlo.reshape %arg4 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %42 = stablehlo.broadcast_in_dim %41, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<8x14x128x128xf32>
    %43 = stablehlo.add %40, %42 : tensor<8x14x128x128xf32>
    %cst_6 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %44 = stablehlo.reduce(%43 init: %cst_6) applies stablehlo.add across dimensions = [1, 2, 3] : (tensor<8x14x128x128xf32>, tensor<f32>) -> tensor<8xf32>
    %45 = stablehlo.broadcast_in_dim %44, dims = [0] : (tensor<8xf32>) -> tensor<8x1x1x1xf32>
    %cst_7 = stablehlo.constant dense<2.293760e+05> : tensor<f32>
    %46 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<8x1x1x1xf32>
    %47 = stablehlo.divide %45, %46 : tensor<8x1x1x1xf32>
    %48 = stablehlo.broadcast_in_dim %47, dims = [0, 1, 2, 3] : (tensor<8x1x1x1xf32>) -> tensor<8x14x128x128xf32>
    %49 = stablehlo.subtract %43, %48 : tensor<8x14x128x128xf32>
    %50 = chlo.square %49 : tensor<8x14x128x128xf32> -> tensor<8x14x128x128xf32>
    %cst_8 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %51 = stablehlo.reduce(%50 init: %cst_8) applies stablehlo.add across dimensions = [1, 2, 3] : (tensor<8x14x128x128xf32>, tensor<f32>) -> tensor<8xf32>
    %52 = stablehlo.broadcast_in_dim %51, dims = [0] : (tensor<8xf32>) -> tensor<8x1x1x1xf32>
    %cst_9 = stablehlo.constant dense<2.293760e+05> : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<8x1x1x1xf32>
    %54 = stablehlo.divide %52, %53 : tensor<8x1x1x1xf32>
    %55 = stablehlo.broadcast_in_dim %47, dims = [0, 1, 2, 3] : (tensor<8x1x1x1xf32>) -> tensor<8x14x128x128xf32>
    %56 = stablehlo.subtract %43, %55 : tensor<8x14x128x128xf32>
    %cst_10 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %57 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<8x1x1x1xf32>
    %58 = stablehlo.add %54, %57 : tensor<8x1x1x1xf32>
    %59 = stablehlo.sqrt %58 : tensor<8x1x1x1xf32>
    %cst_11 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %60 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<8x1x1x1xf32>
    %61 = stablehlo.divide %60, %59 : tensor<8x1x1x1xf32>
    %62 = stablehlo.broadcast_in_dim %61, dims = [0, 1, 2, 3] : (tensor<8x1x1x1xf32>) -> tensor<8x14x128x128xf32>
    %63 = stablehlo.multiply %56, %62 : tensor<8x14x128x128xf32>
    %64 = stablehlo.broadcast_in_dim %arg11, dims = [1, 2, 3] : (tensor<14x128x128xf32>) -> tensor<1x14x128x128xf32>
    %65 = stablehlo.broadcast_in_dim %64, dims = [0, 1, 2, 3] : (tensor<1x14x128x128xf32>) -> tensor<8x14x128x128xf32>
    %66 = stablehlo.multiply %63, %65 : tensor<8x14x128x128xf32>
    %67 = stablehlo.broadcast_in_dim %arg10, dims = [1, 2, 3] : (tensor<14x128x128xf32>) -> tensor<1x14x128x128xf32>
    %68 = stablehlo.broadcast_in_dim %67, dims = [0, 1, 2, 3] : (tensor<1x14x128x128xf32>) -> tensor<8x14x128x128xf32>
    %69 = stablehlo.add %66, %68 : tensor<8x14x128x128xf32>
    %70 = call @relu(%69) : (tensor<8x14x128x128xf32>) -> tensor<8x14x128x128xf32>
    %71 = stablehlo.convolution(%70, %arg7) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision HIGHEST>, #stablehlo<precision HIGHEST>]} : (tensor<8x14x128x128xf32>, tensor<3x3x128x128xf32>) -> tensor<8x14x128x128xf32>
    %72 = stablehlo.reshape %arg6 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %73 = stablehlo.broadcast_in_dim %72, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<8x14x128x128xf32>
    %74 = stablehlo.add %71, %73 : tensor<8x14x128x128xf32>
    %75 = stablehlo.add %74, %12 : tensor<8x14x128x128xf32>
    %cst_12 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %76 = stablehlo.reduce(%75 init: %cst_12) applies stablehlo.add across dimensions = [1, 2, 3] : (tensor<8x14x128x128xf32>, tensor<f32>) -> tensor<8xf32>
    %77 = stablehlo.broadcast_in_dim %76, dims = [0] : (tensor<8xf32>) -> tensor<8x1x1x1xf32>
    %cst_13 = stablehlo.constant dense<2.293760e+05> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<8x1x1x1xf32>
    %79 = stablehlo.divide %77, %78 : tensor<8x1x1x1xf32>
    %80 = stablehlo.broadcast_in_dim %79, dims = [0, 1, 2, 3] : (tensor<8x1x1x1xf32>) -> tensor<8x14x128x128xf32>
    %81 = stablehlo.subtract %75, %80 : tensor<8x14x128x128xf32>
    %82 = chlo.square %81 : tensor<8x14x128x128xf32> -> tensor<8x14x128x128xf32>
    %cst_14 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %83 = stablehlo.reduce(%82 init: %cst_14) applies stablehlo.add across dimensions = [1, 2, 3] : (tensor<8x14x128x128xf32>, tensor<f32>) -> tensor<8xf32>
    %84 = stablehlo.broadcast_in_dim %83, dims = [0] : (tensor<8xf32>) -> tensor<8x1x1x1xf32>
    %cst_15 = stablehlo.constant dense<2.293760e+05> : tensor<f32>
    %85 = stablehlo.broadcast_in_dim %cst_15, dims = [] : (tensor<f32>) -> tensor<8x1x1x1xf32>
    %86 = stablehlo.divide %84, %85 : tensor<8x1x1x1xf32>
    %87 = stablehlo.broadcast_in_dim %79, dims = [0, 1, 2, 3] : (tensor<8x1x1x1xf32>) -> tensor<8x14x128x128xf32>
    %88 = stablehlo.subtract %75, %87 : tensor<8x14x128x128xf32>
    %cst_16 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %89 = stablehlo.broadcast_in_dim %cst_16, dims = [] : (tensor<f32>) -> tensor<8x1x1x1xf32>
    %90 = stablehlo.add %86, %89 : tensor<8x1x1x1xf32>
    %91 = stablehlo.sqrt %90 : tensor<8x1x1x1xf32>
    %cst_17 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %92 = stablehlo.broadcast_in_dim %cst_17, dims = [] : (tensor<f32>) -> tensor<8x1x1x1xf32>
    %93 = stablehlo.divide %92, %91 : tensor<8x1x1x1xf32>
    %94 = stablehlo.broadcast_in_dim %93, dims = [0, 1, 2, 3] : (tensor<8x1x1x1xf32>) -> tensor<8x14x128x128xf32>
    %95 = stablehlo.multiply %88, %94 : tensor<8x14x128x128xf32>
    %96 = stablehlo.broadcast_in_dim %arg17, dims = [1, 2, 3] : (tensor<14x128x128xf32>) -> tensor<1x14x128x128xf32>
    %97 = stablehlo.broadcast_in_dim %96, dims = [0, 1, 2, 3] : (tensor<1x14x128x128xf32>) -> tensor<8x14x128x128xf32>
    %98 = stablehlo.multiply %95, %97 : tensor<8x14x128x128xf32>
    %99 = stablehlo.broadcast_in_dim %arg16, dims = [1, 2, 3] : (tensor<14x128x128xf32>) -> tensor<1x14x128x128xf32>
    %100 = stablehlo.broadcast_in_dim %99, dims = [0, 1, 2, 3] : (tensor<1x14x128x128xf32>) -> tensor<8x14x128x128xf32>
    %101 = stablehlo.add %98, %100 : tensor<8x14x128x128xf32>
    %102 = call @relu(%101) : (tensor<8x14x128x128xf32>) -> tensor<8x14x128x128xf32>
    %103 = stablehlo.convolution(%102, %arg13) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision HIGHEST>, #stablehlo<precision HIGHEST>]} : (tensor<8x14x128x128xf32>, tensor<3x3x128x128xf32>) -> tensor<8x14x128x128xf32>
    %104 = stablehlo.reshape %arg12 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %105 = stablehlo.broadcast_in_dim %104, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<8x14x128x128xf32>
    %106 = stablehlo.add %103, %105 : tensor<8x14x128x128xf32>
    %cst_18 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %107 = stablehlo.reduce(%106 init: %cst_18) applies stablehlo.add across dimensions = [1, 2, 3] : (tensor<8x14x128x128xf32>, tensor<f32>) -> tensor<8xf32>
    %108 = stablehlo.broadcast_in_dim %107, dims = [0] : (tensor<8xf32>) -> tensor<8x1x1x1xf32>
    %cst_19 = stablehlo.constant dense<2.293760e+05> : tensor<f32>
    %109 = stablehlo.broadcast_in_dim %cst_19, dims = [] : (tensor<f32>) -> tensor<8x1x1x1xf32>
    %110 = stablehlo.divide %108, %109 : tensor<8x1x1x1xf32>
    %111 = stablehlo.broadcast_in_dim %110, dims = [0, 1, 2, 3] : (tensor<8x1x1x1xf32>) -> tensor<8x14x128x128xf32>
    %112 = stablehlo.subtract %106, %111 : tensor<8x14x128x128xf32>
    %113 = chlo.square %112 : tensor<8x14x128x128xf32> -> tensor<8x14x128x128xf32>
    %cst_20 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %114 = stablehlo.reduce(%113 init: %cst_20) applies stablehlo.add across dimensions = [1, 2, 3] : (tensor<8x14x128x128xf32>, tensor<f32>) -> tensor<8xf32>
    %115 = stablehlo.broadcast_in_dim %114, dims = [0] : (tensor<8xf32>) -> tensor<8x1x1x1xf32>
    %cst_21 = stablehlo.constant dense<2.293760e+05> : tensor<f32>
    %116 = stablehlo.broadcast_in_dim %cst_21, dims = [] : (tensor<f32>) -> tensor<8x1x1x1xf32>
    %117 = stablehlo.divide %115, %116 : tensor<8x1x1x1xf32>
    %118 = stablehlo.broadcast_in_dim %110, dims = [0, 1, 2, 3] : (tensor<8x1x1x1xf32>) -> tensor<8x14x128x128xf32>
    %119 = stablehlo.subtract %106, %118 : tensor<8x14x128x128xf32>
    %cst_22 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %120 = stablehlo.broadcast_in_dim %cst_22, dims = [] : (tensor<f32>) -> tensor<8x1x1x1xf32>
    %121 = stablehlo.add %117, %120 : tensor<8x1x1x1xf32>
    %122 = stablehlo.sqrt %121 : tensor<8x1x1x1xf32>
    %cst_23 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %123 = stablehlo.broadcast_in_dim %cst_23, dims = [] : (tensor<f32>) -> tensor<8x1x1x1xf32>
    %124 = stablehlo.divide %123, %122 : tensor<8x1x1x1xf32>
    %125 = stablehlo.broadcast_in_dim %124, dims = [0, 1, 2, 3] : (tensor<8x1x1x1xf32>) -> tensor<8x14x128x128xf32>
    %126 = stablehlo.multiply %119, %125 : tensor<8x14x128x128xf32>
    %127 = stablehlo.broadcast_in_dim %arg19, dims = [1, 2, 3] : (tensor<14x128x128xf32>) -> tensor<1x14x128x128xf32>
    %128 = stablehlo.broadcast_in_dim %127, dims = [0, 1, 2, 3] : (tensor<1x14x128x128xf32>) -> tensor<8x14x128x128xf32>
    %129 = stablehlo.multiply %126, %128 : tensor<8x14x128x128xf32>
    %130 = stablehlo.broadcast_in_dim %arg18, dims = [1, 2, 3] : (tensor<14x128x128xf32>) -> tensor<1x14x128x128xf32>
    %131 = stablehlo.broadcast_in_dim %130, dims = [0, 1, 2, 3] : (tensor<1x14x128x128xf32>) -> tensor<8x14x128x128xf32>
    %132 = stablehlo.add %129, %131 : tensor<8x14x128x128xf32>
    %133 = call @relu(%132) : (tensor<8x14x128x128xf32>) -> tensor<8x14x128x128xf32>
    %134 = stablehlo.convolution(%133, %arg15) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision HIGHEST>, #stablehlo<precision HIGHEST>]} : (tensor<8x14x128x128xf32>, tensor<3x3x128x128xf32>) -> tensor<8x14x128x128xf32>
    %135 = stablehlo.reshape %arg14 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %136 = stablehlo.broadcast_in_dim %135, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<8x14x128x128xf32>
    %137 = stablehlo.add %134, %136 : tensor<8x14x128x128xf32>
    %138 = stablehlo.add %137, %75 : tensor<8x14x128x128xf32>
    %cst_24 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %139 = stablehlo.reduce(%138 init: %cst_24) applies stablehlo.add across dimensions = [1, 2, 3] : (tensor<8x14x128x128xf32>, tensor<f32>) -> tensor<8xf32>
    %140 = stablehlo.broadcast_in_dim %139, dims = [0] : (tensor<8xf32>) -> tensor<8x1x1x1xf32>
    %cst_25 = stablehlo.constant dense<2.293760e+05> : tensor<f32>
    %141 = stablehlo.broadcast_in_dim %cst_25, dims = [] : (tensor<f32>) -> tensor<8x1x1x1xf32>
    %142 = stablehlo.divide %140, %141 : tensor<8x1x1x1xf32>
    %143 = stablehlo.broadcast_in_dim %142, dims = [0, 1, 2, 3] : (tensor<8x1x1x1xf32>) -> tensor<8x14x128x128xf32>
    %144 = stablehlo.subtract %138, %143 : tensor<8x14x128x128xf32>
    %145 = chlo.square %144 : tensor<8x14x128x128xf32> -> tensor<8x14x128x128xf32>
    %cst_26 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %146 = stablehlo.reduce(%145 init: %cst_26) applies stablehlo.add across dimensions = [1, 2, 3] : (tensor<8x14x128x128xf32>, tensor<f32>) -> tensor<8xf32>
    %147 = stablehlo.broadcast_in_dim %146, dims = [0] : (tensor<8xf32>) -> tensor<8x1x1x1xf32>
    %cst_27 = stablehlo.constant dense<2.293760e+05> : tensor<f32>
    %148 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<8x1x1x1xf32>
    %149 = stablehlo.divide %147, %148 : tensor<8x1x1x1xf32>
    %150 = stablehlo.broadcast_in_dim %142, dims = [0, 1, 2, 3] : (tensor<8x1x1x1xf32>) -> tensor<8x14x128x128xf32>
    %151 = stablehlo.subtract %138, %150 : tensor<8x14x128x128xf32>
    %cst_28 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %152 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<8x1x1x1xf32>
    %153 = stablehlo.add %149, %152 : tensor<8x1x1x1xf32>
    %154 = stablehlo.sqrt %153 : tensor<8x1x1x1xf32>
    %cst_29 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %155 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<8x1x1x1xf32>
    %156 = stablehlo.divide %155, %154 : tensor<8x1x1x1xf32>
    %157 = stablehlo.broadcast_in_dim %156, dims = [0, 1, 2, 3] : (tensor<8x1x1x1xf32>) -> tensor<8x14x128x128xf32>
    %158 = stablehlo.multiply %151, %157 : tensor<8x14x128x128xf32>
    %159 = stablehlo.broadcast_in_dim %arg25, dims = [1, 2, 3] : (tensor<14x128x128xf32>) -> tensor<1x14x128x128xf32>
    %160 = stablehlo.broadcast_in_dim %159, dims = [0, 1, 2, 3] : (tensor<1x14x128x128xf32>) -> tensor<8x14x128x128xf32>
    %161 = stablehlo.multiply %158, %160 : tensor<8x14x128x128xf32>
    %162 = stablehlo.broadcast_in_dim %arg24, dims = [1, 2, 3] : (tensor<14x128x128xf32>) -> tensor<1x14x128x128xf32>
    %163 = stablehlo.broadcast_in_dim %162, dims = [0, 1, 2, 3] : (tensor<1x14x128x128xf32>) -> tensor<8x14x128x128xf32>
    %164 = stablehlo.add %161, %163 : tensor<8x14x128x128xf32>
    %165 = call @relu(%164) : (tensor<8x14x128x128xf32>) -> tensor<8x14x128x128xf32>
    %166 = stablehlo.convolution(%165, %arg21) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision HIGHEST>, #stablehlo<precision HIGHEST>]} : (tensor<8x14x128x128xf32>, tensor<3x3x128x128xf32>) -> tensor<8x14x128x128xf32>
    %167 = stablehlo.reshape %arg20 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %168 = stablehlo.broadcast_in_dim %167, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<8x14x128x128xf32>
    %169 = stablehlo.add %166, %168 : tensor<8x14x128x128xf32>
    %cst_30 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %170 = stablehlo.reduce(%169 init: %cst_30) applies stablehlo.add across dimensions = [1, 2, 3] : (tensor<8x14x128x128xf32>, tensor<f32>) -> tensor<8xf32>
    %171 = stablehlo.broadcast_in_dim %170, dims = [0] : (tensor<8xf32>) -> tensor<8x1x1x1xf32>
    %cst_31 = stablehlo.constant dense<2.293760e+05> : tensor<f32>
    %172 = stablehlo.broadcast_in_dim %cst_31, dims = [] : (tensor<f32>) -> tensor<8x1x1x1xf32>
    %173 = stablehlo.divide %171, %172 : tensor<8x1x1x1xf32>
    %174 = stablehlo.broadcast_in_dim %173, dims = [0, 1, 2, 3] : (tensor<8x1x1x1xf32>) -> tensor<8x14x128x128xf32>
    %175 = stablehlo.subtract %169, %174 : tensor<8x14x128x128xf32>
    %176 = chlo.square %175 : tensor<8x14x128x128xf32> -> tensor<8x14x128x128xf32>
    %cst_32 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %177 = stablehlo.reduce(%176 init: %cst_32) applies stablehlo.add across dimensions = [1, 2, 3] : (tensor<8x14x128x128xf32>, tensor<f32>) -> tensor<8xf32>
    %178 = stablehlo.broadcast_in_dim %177, dims = [0] : (tensor<8xf32>) -> tensor<8x1x1x1xf32>
    %cst_33 = stablehlo.constant dense<2.293760e+05> : tensor<f32>
    %179 = stablehlo.broadcast_in_dim %cst_33, dims = [] : (tensor<f32>) -> tensor<8x1x1x1xf32>
    %180 = stablehlo.divide %178, %179 : tensor<8x1x1x1xf32>
    %181 = stablehlo.broadcast_in_dim %173, dims = [0, 1, 2, 3] : (tensor<8x1x1x1xf32>) -> tensor<8x14x128x128xf32>
    %182 = stablehlo.subtract %169, %181 : tensor<8x14x128x128xf32>
    %cst_34 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %183 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<8x1x1x1xf32>
    %184 = stablehlo.add %180, %183 : tensor<8x1x1x1xf32>
    %185 = stablehlo.sqrt %184 : tensor<8x1x1x1xf32>
    %cst_35 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %186 = stablehlo.broadcast_in_dim %cst_35, dims = [] : (tensor<f32>) -> tensor<8x1x1x1xf32>
    %187 = stablehlo.divide %186, %185 : tensor<8x1x1x1xf32>
    %188 = stablehlo.broadcast_in_dim %187, dims = [0, 1, 2, 3] : (tensor<8x1x1x1xf32>) -> tensor<8x14x128x128xf32>
    %189 = stablehlo.multiply %182, %188 : tensor<8x14x128x128xf32>
    %190 = stablehlo.broadcast_in_dim %arg27, dims = [1, 2, 3] : (tensor<14x128x128xf32>) -> tensor<1x14x128x128xf32>
    %191 = stablehlo.broadcast_in_dim %190, dims = [0, 1, 2, 3] : (tensor<1x14x128x128xf32>) -> tensor<8x14x128x128xf32>
    %192 = stablehlo.multiply %189, %191 : tensor<8x14x128x128xf32>
    %193 = stablehlo.broadcast_in_dim %arg26, dims = [1, 2, 3] : (tensor<14x128x128xf32>) -> tensor<1x14x128x128xf32>
    %194 = stablehlo.broadcast_in_dim %193, dims = [0, 1, 2, 3] : (tensor<1x14x128x128xf32>) -> tensor<8x14x128x128xf32>
    %195 = stablehlo.add %192, %194 : tensor<8x14x128x128xf32>
    %196 = call @relu(%195) : (tensor<8x14x128x128xf32>) -> tensor<8x14x128x128xf32>
    %197 = stablehlo.convolution(%196, %arg23) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision HIGHEST>, #stablehlo<precision HIGHEST>]} : (tensor<8x14x128x128xf32>, tensor<3x3x128x128xf32>) -> tensor<8x14x128x128xf32>
    %198 = stablehlo.reshape %arg22 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %199 = stablehlo.broadcast_in_dim %198, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<8x14x128x128xf32>
    %200 = stablehlo.add %197, %199 : tensor<8x14x128x128xf32>
    %201 = stablehlo.add %200, %138 : tensor<8x14x128x128xf32>
    %cst_36 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %202 = stablehlo.reduce(%201 init: %cst_36) applies stablehlo.add across dimensions = [1, 2, 3] : (tensor<8x14x128x128xf32>, tensor<f32>) -> tensor<8xf32>
    %203 = stablehlo.broadcast_in_dim %202, dims = [0] : (tensor<8xf32>) -> tensor<8x1x1x1xf32>
    %cst_37 = stablehlo.constant dense<2.293760e+05> : tensor<f32>
    %204 = stablehlo.broadcast_in_dim %cst_37, dims = [] : (tensor<f32>) -> tensor<8x1x1x1xf32>
    %205 = stablehlo.divide %203, %204 : tensor<8x1x1x1xf32>
    %206 = stablehlo.broadcast_in_dim %205, dims = [0, 1, 2, 3] : (tensor<8x1x1x1xf32>) -> tensor<8x14x128x128xf32>
    %207 = stablehlo.subtract %201, %206 : tensor<8x14x128x128xf32>
    %208 = chlo.square %207 : tensor<8x14x128x128xf32> -> tensor<8x14x128x128xf32>
    %cst_38 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %209 = stablehlo.reduce(%208 init: %cst_38) applies stablehlo.add across dimensions = [1, 2, 3] : (tensor<8x14x128x128xf32>, tensor<f32>) -> tensor<8xf32>
    %210 = stablehlo.broadcast_in_dim %209, dims = [0] : (tensor<8xf32>) -> tensor<8x1x1x1xf32>
    %cst_39 = stablehlo.constant dense<2.293760e+05> : tensor<f32>
    %211 = stablehlo.broadcast_in_dim %cst_39, dims = [] : (tensor<f32>) -> tensor<8x1x1x1xf32>
    %212 = stablehlo.divide %210, %211 : tensor<8x1x1x1xf32>
    %213 = stablehlo.broadcast_in_dim %205, dims = [0, 1, 2, 3] : (tensor<8x1x1x1xf32>) -> tensor<8x14x128x128xf32>
    %214 = stablehlo.subtract %201, %213 : tensor<8x14x128x128xf32>
    %cst_40 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %215 = stablehlo.broadcast_in_dim %cst_40, dims = [] : (tensor<f32>) -> tensor<8x1x1x1xf32>
    %216 = stablehlo.add %212, %215 : tensor<8x1x1x1xf32>
    %217 = stablehlo.sqrt %216 : tensor<8x1x1x1xf32>
    %cst_41 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %218 = stablehlo.broadcast_in_dim %cst_41, dims = [] : (tensor<f32>) -> tensor<8x1x1x1xf32>
    %219 = stablehlo.divide %218, %217 : tensor<8x1x1x1xf32>
    %220 = stablehlo.broadcast_in_dim %219, dims = [0, 1, 2, 3] : (tensor<8x1x1x1xf32>) -> tensor<8x14x128x128xf32>
    %221 = stablehlo.multiply %214, %220 : tensor<8x14x128x128xf32>
    %222 = stablehlo.broadcast_in_dim %arg33, dims = [1, 2, 3] : (tensor<14x128x128xf32>) -> tensor<1x14x128x128xf32>
    %223 = stablehlo.broadcast_in_dim %222, dims = [0, 1, 2, 3] : (tensor<1x14x128x128xf32>) -> tensor<8x14x128x128xf32>
    %224 = stablehlo.multiply %221, %223 : tensor<8x14x128x128xf32>
    %225 = stablehlo.broadcast_in_dim %arg32, dims = [1, 2, 3] : (tensor<14x128x128xf32>) -> tensor<1x14x128x128xf32>
    %226 = stablehlo.broadcast_in_dim %225, dims = [0, 1, 2, 3] : (tensor<1x14x128x128xf32>) -> tensor<8x14x128x128xf32>
    %227 = stablehlo.add %224, %226 : tensor<8x14x128x128xf32>
    %228 = call @relu(%227) : (tensor<8x14x128x128xf32>) -> tensor<8x14x128x128xf32>
    %229 = stablehlo.convolution(%228, %arg29) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision HIGHEST>, #stablehlo<precision HIGHEST>]} : (tensor<8x14x128x128xf32>, tensor<3x3x128x128xf32>) -> tensor<8x14x128x128xf32>
    %230 = stablehlo.reshape %arg28 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %231 = stablehlo.broadcast_in_dim %230, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<8x14x128x128xf32>
    %232 = stablehlo.add %229, %231 : tensor<8x14x128x128xf32>
    %cst_42 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %233 = stablehlo.reduce(%232 init: %cst_42) applies stablehlo.add across dimensions = [1, 2, 3] : (tensor<8x14x128x128xf32>, tensor<f32>) -> tensor<8xf32>
    %234 = stablehlo.broadcast_in_dim %233, dims = [0] : (tensor<8xf32>) -> tensor<8x1x1x1xf32>
    %cst_43 = stablehlo.constant dense<2.293760e+05> : tensor<f32>
    %235 = stablehlo.broadcast_in_dim %cst_43, dims = [] : (tensor<f32>) -> tensor<8x1x1x1xf32>
    %236 = stablehlo.divide %234, %235 : tensor<8x1x1x1xf32>
    %237 = stablehlo.broadcast_in_dim %236, dims = [0, 1, 2, 3] : (tensor<8x1x1x1xf32>) -> tensor<8x14x128x128xf32>
    %238 = stablehlo.subtract %232, %237 : tensor<8x14x128x128xf32>
    %239 = chlo.square %238 : tensor<8x14x128x128xf32> -> tensor<8x14x128x128xf32>
    %cst_44 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %240 = stablehlo.reduce(%239 init: %cst_44) applies stablehlo.add across dimensions = [1, 2, 3] : (tensor<8x14x128x128xf32>, tensor<f32>) -> tensor<8xf32>
    %241 = stablehlo.broadcast_in_dim %240, dims = [0] : (tensor<8xf32>) -> tensor<8x1x1x1xf32>
    %cst_45 = stablehlo.constant dense<2.293760e+05> : tensor<f32>
    %242 = stablehlo.broadcast_in_dim %cst_45, dims = [] : (tensor<f32>) -> tensor<8x1x1x1xf32>
    %243 = stablehlo.divide %241, %242 : tensor<8x1x1x1xf32>
    %244 = stablehlo.broadcast_in_dim %236, dims = [0, 1, 2, 3] : (tensor<8x1x1x1xf32>) -> tensor<8x14x128x128xf32>
    %245 = stablehlo.subtract %232, %244 : tensor<8x14x128x128xf32>
    %cst_46 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %246 = stablehlo.broadcast_in_dim %cst_46, dims = [] : (tensor<f32>) -> tensor<8x1x1x1xf32>
    %247 = stablehlo.add %243, %246 : tensor<8x1x1x1xf32>
    %248 = stablehlo.sqrt %247 : tensor<8x1x1x1xf32>
    %cst_47 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %249 = stablehlo.broadcast_in_dim %cst_47, dims = [] : (tensor<f32>) -> tensor<8x1x1x1xf32>
    %250 = stablehlo.divide %249, %248 : tensor<8x1x1x1xf32>
    %251 = stablehlo.broadcast_in_dim %250, dims = [0, 1, 2, 3] : (tensor<8x1x1x1xf32>) -> tensor<8x14x128x128xf32>
    %252 = stablehlo.multiply %245, %251 : tensor<8x14x128x128xf32>
    %253 = stablehlo.broadcast_in_dim %arg35, dims = [1, 2, 3] : (tensor<14x128x128xf32>) -> tensor<1x14x128x128xf32>
    %254 = stablehlo.broadcast_in_dim %253, dims = [0, 1, 2, 3] : (tensor<1x14x128x128xf32>) -> tensor<8x14x128x128xf32>
    %255 = stablehlo.multiply %252, %254 : tensor<8x14x128x128xf32>
    %256 = stablehlo.broadcast_in_dim %arg34, dims = [1, 2, 3] : (tensor<14x128x128xf32>) -> tensor<1x14x128x128xf32>
    %257 = stablehlo.broadcast_in_dim %256, dims = [0, 1, 2, 3] : (tensor<1x14x128x128xf32>) -> tensor<8x14x128x128xf32>
    %258 = stablehlo.add %255, %257 : tensor<8x14x128x128xf32>
    %259 = call @relu(%258) : (tensor<8x14x128x128xf32>) -> tensor<8x14x128x128xf32>
    %260 = stablehlo.convolution(%259, %arg31) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision HIGHEST>, #stablehlo<precision HIGHEST>]} : (tensor<8x14x128x128xf32>, tensor<3x3x128x128xf32>) -> tensor<8x14x128x128xf32>
    %261 = stablehlo.reshape %arg30 : (tensor<128xf32>) -> tensor<1x1x1x128xf32>
    %262 = stablehlo.broadcast_in_dim %261, dims = [0, 1, 2, 3] : (tensor<1x1x1x128xf32>) -> tensor<8x14x128x128xf32>
    %263 = stablehlo.add %260, %262 : tensor<8x14x128x128xf32>
    %264 = stablehlo.add %263, %201 : tensor<8x14x128x128xf32>
    %265 = stablehlo.convolution(%264, %arg3) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision HIGHEST>, #stablehlo<precision HIGHEST>]} : (tensor<8x14x128x128xf32>, tensor<3x3x128x2xf32>) -> tensor<8x14x128x2xf32>
    %266 = stablehlo.reshape %arg2 : (tensor<2xf32>) -> tensor<1x1x1x2xf32>
    %267 = stablehlo.broadcast_in_dim %266, dims = [0, 1, 2, 3] : (tensor<1x1x1x2xf32>) -> tensor<8x14x128x2xf32>
    %268 = stablehlo.add %265, %267 : tensor<8x14x128x2xf32>
    return %268 : tensor<8x14x128x2xf32>
  }
  func.func private @relu(%arg0: tensor<8x14x128x128xf32>) -> tensor<8x14x128x128xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x14x128x128xf32>
    %1 = stablehlo.maximum %arg0, %0 : tensor<8x14x128x128xf32>
    return %1 : tensor<8x14x128x128xf32>
  }
}
