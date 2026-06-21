# npp-rs shape survey — NPP 12.4.1.87 (CUDA 12.9)

Produced by `cargo run -p npp-codegen` on 2026-06-21. The coverage curve
determines the codegen strategy (Phase 1.4). The shape histogram head lists
the exact macros to write. The singleton tail sizes the un-automated risk.

> **F2 selection:** Three families were implemented from this survey — **Resize**
> (top-2 shape, 27 functions), **SwapChannels** (shape `SRC+STEP, DST+STEP, SIZE,
> CHANNEL_ORDER`, 20 functions), and **Mean** (shape `SRC+STEP, SIZE, ptr:dst,
> ptr:dst`, 16 functions). See `docs/codegen-architecture.md` for the generation
> flow and `npp-codegen/src/gen_impls.rs` for their `FamilyDescriptor` definitions.

== TOTALS ==
distinct functions (base, _Ctx collapsed) : 5606
  ...of which have a _Ctx twin            : 5515
distinct families                         : 384
distinct shapes                           : 348

== COVERAGE CURVE ==
  top  5: 1631 / 5606  (29.1%)
  top 10: 2132 / 5606  (38.0%)
  top 15: 2552 / 5606  (45.5%)
  top 20: 2854 / 5606  (50.9%)
  top 30: 3296 / 5606  (58.8%)

== SINGLETON TAIL ==
  shapes used by exactly 1 function: 105
  shapes used by exactly 2 functions: 44
  shapes used by exactly 3 functions: 21

== SHAPE HISTOGRAM ==
  686 | 14 | 686
  449 | 24 | 449
  270 | 60 | 270
  113 | 32 | 113
  113 | 49 | 113
  108 | 48 | 108
  106 | 82 | 106
  96 | 56 | 96
  96 | 46 | 96
  95 | 33 | 95
  93 | 42 | 93
  93 | 32 | 93
  80 | 52 | 80
  77 | 101 | 77
  77 | 67 | 77
  68 | 63 | 68
  60 | 34 | 60
  60 | 58 | 60
  57 | 47 | 57
  57 | 66 | 57
  49 | 61 | 49
  48 | 62 | 48
  48 | 95 | 48
  45 | 46 | 45
  45 | 80 | 45
  45 | 67 | 45
  44 | 34 | 44
  40 | 50 | 40
  40 | 89 | 40
  38 | 64 | 38
  38 | 34 | 38
  36 | 54 | 36
  36 | 53 | 36
  34 | 83 | 34
  33 | 38 | 33
  33 | 37 | 33
  32 | 52 | 32
  32 | 73 | 32
  31 | 23 | 31
  30 | 28 | 30
  30 | 50 | 30
  30 | 23 | 30
  29 | 54 | 29
  28 | 50 | 28
  28 | 24 | 28
  28 | 33 | 28
  28 | 104 | 28
  27 | 43 | 27
  27 | 50 | 27
  24 | 68 | 24
  24 | 48 | 24
  24 | 93 | 24
  24 | 158 | 24
  24 | 63 | 24
  24 | 130 | 24
  24 | 96 | 24
  24 | 89 | 24
  24 | 79 | 24
  23 | 33 | 23
  22 | 47 | 22
  21 | 104 | 21
  20 | 43 | 20
  20 | 43 | 20
  20 | 39 | 20
  20 | 34 | 20
  20 | 41 | 20
  20 | 38 | 20
  20 | 33 | 20
  20 | 93 | 20
  20 | 91 | 20
  18 | 108 | 18
  18 | 74 | 18
  18 | 70 | 18
  17 | 57 | 17
  16 | 34 | 16
  16 | 58 | 16
  16 | 48 | 16
  16 | 74 | 16
  16 | 19 | 16
  16 | 40 | 16
  16 | 42 | 16
  16 | 61 | 16
  16 | 51 | 16
  16 | 32 | 16
  16 | 41 | 16
  16 | 137 | 16
  16 | 101 | 16
  16 | 67 | 16
  16 | 70 | 16
  16 | 67 | 16
  15 | 59 | 15
  15 | 42 | 15
  14 | 53 | 14
  13 | 24 | 13
  13 | 36 | 13
  13 | 104 | 13
  12 | 24 | 12
  12 | 44 | 12
  12 | 14 | 12
  12 | 47 | 12
  12 | 52 | 12
  12 | 53 | 12
  12 | 58 | 12
  12 | 72 | 12
  12 | 66 | 12
  12 | 48 | 12
  12 | 51 | 12
  12 | 114 | 12
  12 | 117 | 12
  12 | 83 | 12
  12 | 85 | 12
  12 | 70 | 12
  12 | 83 | 12
  11 | 69 | 11
  11 | 52 | 11
  10 | 34 | 10
  10 | 94 | 10
  10 | 92 | 10
  10 | 33 | 10
  9 | 34 | 9
  9 | 61 | 9
  8 | 36 | 8
  8 | 36 | 8
  8 | 35 | 8
  8 | 39 | 8
  8 | 53 | 8
  8 | 48 | 8
  8 | 57 | 8
  8 | 60 | 8
  8 | 51 | 8
  8 | 106 | 8
  8 | 47 | 8
  8 | 56 | 8
  8 | 59 | 8
  8 | 117 | 8
  8 | 46 | 8
  8 | 46 | 8
  8 | 45 | 8
  6 | 49 | 6
  6 | 24 | 6
  6 | 44 | 6
  6 | 91 | 6
  6 | 100 | 6
  6 | 128 | 6
  4 | 50 | 4
  4 | 52 | 4
  4 | 50 | 4
  4 | 27 | 4
  4 | 37 | 4
  4 | 36 | 4
  4 | 38 | 4
  4 | 50 | 4
  4 | 52 | 4
  4 | 49 | 4
  4 | 62 | 4
  4 | 47 | 4
  4 | 48 | 4
  4 | 62 | 4
  4 | 39 | 4
  4 | 125 | 4
  4 | 125 | 4
  4 | 123 | 4
  4 | 123 | 4
  4 | 71 | 4
  4 | 61 | 4
  4 | 51 | 4
  4 | 90 | 4
  4 | 70 | 4
  4 | 80 | 4
  4 | 60 | 4
  4 | 60 | 4
  4 | 60 | 4
  4 | 96 | 4
  4 | 46 | 4
  4 | 60 | 4
  4 | 59 | 4
  4 | 94 | 4
  4 | 65 | 4
  3 | 27 | 3
  3 | 27 | 3
  3 | 27 | 3
  3 | 36 | 3
  3 | 26 | 3
  3 | 40 | 3
  3 | 37 | 3
  3 | 48 | 3
  3 | 43 | 3
  3 | 86 | 3
  3 | 68 | 3
  3 | 46 | 3
  3 | 46 | 3
  3 | 52 | 3
  3 | 65 | 3
  3 | 73 | 3
  3 | 82 | 3
  3 | 42 | 3
  3 | 46 | 3
  3 | 61 | 3
  3 | 80 | 3
  2 | 21 | 2
  2 | 54 | 2
  2 | 74 | 2
  2 | 44 | 2
  2 | 24 | 2
  2 | 58 | 2
  2 | 37 | 2
  2 | 50 | 2
  2 | 37 | 2
  2 | 50 | 2
  2 | 48 | 2
  2 | 47 | 2
  2 | 46 | 2
  2 | 53 | 2
  2 | 134 | 2
  2 | 134 | 2
  2 | 138 | 2
  2 | 134 | 2
  2 | 147 | 2
  2 | 147 | 2
  2 | 134 | 2
  2 | 132 | 2
  2 | 132 | 2
  2 | 33 | 2
  2 | 49 | 2
  2 | 201 | 2
  2 | 111 | 2
  2 | 116 | 2
  2 | 103 | 2
  2 | 116 | 2
  2 | 103 | 2
  2 | 112 | 2
  2 | 100 | 2
  2 | 36 | 2
  2 | 49 | 2
  2 | 36 | 2
  2 | 49 | 2
  2 | 36 | 2
  2 | 49 | 2
  2 | 35 | 2
  2 | 47 | 2
  2 | 42 | 2
  2 | 33 | 2
  2 | 61 | 2
  1 | 46 | 1
  1 | 28 | 1
  1 | 46 | 1
  1 | 46 | 1
  1 | 28 | 1
  1 | 46 | 1
  1 | 28 | 1
  1 | 27 | 1
  1 | 20 | 1
  1 | 26 | 1
  1 | 45 | 1
  1 | 111 | 1
  1 | 101 | 1
  1 | 38 | 1
  1 | 54 | 1
  1 | 24 | 1
  1 | 18 | 1
  1 | 41 | 1
  1 | 51 | 1
  1 | 8 | 1
  1 | 64 | 1
  1 | 43 | 1
  1 | 219 | 1
  1 | 62 | 1
  1 | 20 | 1
  1 | 44 | 1
  1 | 28 | 1
  1 | 76 | 1
  1 | 65 | 1
  1 | 52 | 1
  1 | 76 | 1
  1 | 65 | 1
  1 | 52 | 1
  1 | 76 | 1
  1 | 65 | 1
  1 | 52 | 1
  1 | 37 | 1
  1 | 72 | 1
  1 | 63 | 1
  1 | 51 | 1
  1 | 47 | 1
  1 | 47 | 1
  1 | 47 | 1
  1 | 47 | 1
  1 | 46 | 1
  1 | 69 | 1
  1 | 69 | 1
  1 | 69 | 1
  1 | 52 | 1
  1 | 52 | 1
  1 | 50 | 1
  1 | 52 | 1
  1 | 51 | 1
  1 | 63 | 1
  1 | 63 | 1
  1 | 63 | 1
  1 | 63 | 1
  1 | 62 | 1
  1 | 83 | 1
  1 | 93 | 1
  1 | 82 | 1
  1 | 56 | 1
  1 | 59 | 1
  1 | 57 | 1
  1 | 156 | 1
  1 | 160 | 1
  1 | 99 | 1
  1 | 99 | 1
  1 | 98 | 1
  1 | 63 | 1
  1 | 140 | 1
  1 | 68 | 1
  1 | 43 | 1
  1 | 90 | 1
  1 | 77 | 1
  1 | 90 | 1
  1 | 77 | 1
  1 | 88 | 1
  1 | 76 | 1
  1 | 75 | 1
  1 | 64 | 1
  1 | 51 | 1
  1 | 75 | 1
  1 | 64 | 1
  1 | 51 | 1
  1 | 75 | 1
  1 | 64 | 1
  1 | 51 | 1
  1 | 71 | 1
  1 | 62 | 1
  1 | 50 | 1
  1 | 52 | 1
  1 | 52 | 1
  1 | 125 | 1
  1 | 46 | 1
  1 | 46 | 1
  1 | 45 | 1
  1 | 96 | 1
  1 | 96 | 1
  1 | 96 | 1
  1 | 96 | 1
  1 | 94 | 1
  1 | 103 | 1
  1 | 64 | 1
  1 | 82 | 1

== FAMILY→SHAPE TABLE (families using each shape) ==
  MISC:Npp16s, DST+STEP, SIZE:
    - nppiSet_ (3)
  MISC:Npp16s, DST+STEP, SIZE, ptr:src, MISC:i32:
    - nppiSet_ (1)
  MISC:Npp16s, ptr:dst, MISC:i32, SIZE, CONST_SCALAR:
    - nppiAddC_ (1)
    - nppiDivC_ (1)
    - nppiMulC_ (1)
    - nppiSubC_ (1)
  MISC:Npp16sc, DST+STEP, SIZE:
    - nppiSet_ (1)
  MISC:Npp16sc, MISC:ptr, MISC:i32, SIZE, CONST_SCALAR:
    - nppiAddC_ (1)
    - nppiDivC_ (1)
    - nppiMulC_ (1)
    - nppiSubC_ (1)
  MISC:Npp16u, DST+STEP, SIZE:
    - nppiSet_ (3)
  MISC:Npp16u, DST+STEP, SIZE, ptr:src, MISC:i32:
    - nppiSet_ (1)
  MISC:Npp16u, ptr:dst, MISC:i32, SIZE:
    - nppiAlphaPremulC_ (4)
    - nppiAndC_ (1)
    - nppiMulCScale_ (1)
    - nppiOrC_ (1)
    - nppiXorC_ (1)
  MISC:Npp16u, ptr:dst, MISC:i32, SIZE, CONST_SCALAR:
    - nppiAddC_ (1)
    - nppiDivC_ (1)
    - nppiMulC_ (1)
    - nppiSubC_ (1)
  MISC:Npp32f, DST+STEP, SIZE:
    - nppiSet_ (4)
  MISC:Npp32f, DST+STEP, SIZE, ptr:src, MISC:i32:
    - nppiSet_ (1)
  MISC:Npp32f, MISC:Npp32f, MISC:i32:
    - nppiFilterUnsharpGetBufferSize_ (16)
  MISC:Npp32f, MISC:Npp32f, SIZE, MISC:ptr, MISC:i32:
    - nppiColorTwistBatch (18)
    - nppiColorTwistBatch_ (10)
  MISC:Npp32f, MISC:ptr, MISC:i32, SIZE:
    - nppiAddC_ (1)
    - nppiDivC_ (1)
    - nppiMulC_ (1)
    - nppiSubC_ (1)
  MISC:Npp32f, ptr:dst, MISC:i32, SIZE:
    - nppiAddC_ (1)
    - nppiDivC_ (1)
    - nppiMulC_ (1)
    - nppiSubC_ (1)
  MISC:Npp32fc, DST+STEP, SIZE:
    - nppiSet_ (1)
  MISC:Npp32fc, MISC:ptr, MISC:i32, SIZE:
    - nppiAddC_ (1)
    - nppiDivC_ (1)
    - nppiMulC_ (1)
    - nppiSubC_ (1)
  MISC:Npp32s, DST+STEP, SIZE:
    - nppiSet_ (3)
  MISC:Npp32s, DST+STEP, SIZE, ptr:src, MISC:i32:
    - nppiSet_ (1)
  MISC:Npp32s, ptr:dst, MISC:i32, SIZE:
    - nppiAndC_ (1)
    - nppiOrC_ (1)
    - nppiXorC_ (1)
  MISC:Npp32s, ptr:dst, MISC:i32, SIZE, CONST_SCALAR:
    - nppiAddC_ (1)
    - nppiDivC_ (1)
    - nppiMulC_ (1)
    - nppiSubC_ (1)
  MISC:Npp32sc, DST+STEP, SIZE:
    - nppiSet_ (1)
  MISC:Npp32sc, MISC:ptr, MISC:i32, SIZE, CONST_SCALAR:
    - nppiAddC_ (1)
    - nppiDivC_ (1)
    - nppiMulC_ (1)
    - nppiSubC_ (1)
  MISC:Npp32u, DST+STEP, SIZE:
    - nppiSet_ (1)
  MISC:Npp32u, ptr:dst:
    - nppiCompressedMarkerLabelsUFGetGeometryListsSize_C (1)
  MISC:Npp32u, ptr:dst, MISC:i32, SIZE:
    - nppiLShiftC_ (3)
    - nppiRShiftC_ (5)
  MISC:Npp8s, DST+STEP, SIZE:
    - nppiSet_ (1)
  MISC:Npp8u, DST+STEP, SIZE:
    - nppiSet_ (3)
  MISC:Npp8u, DST+STEP, SIZE, ptr:src, MISC:i32:
    - nppiSet_ (1)
  MISC:Npp8u, ptr:dst, MISC:i32, SIZE:
    - nppiAlphaPremulC_ (4)
    - nppiAndC_ (1)
    - nppiMulCScale_ (1)
    - nppiOrC_ (1)
    - nppiXorC_ (1)
  MISC:Npp8u, ptr:dst, MISC:i32, SIZE, CONST_SCALAR:
    - nppiAddC_ (1)
    - nppiDivC_ (1)
    - nppiMulC_ (1)
    - nppiSubC_ (1)
  MISC:NppDataType, MISC:NppiChannels, MISC:ptr, MISC:i32, MISC:ptr, MISC:i32, DST+STEP, SIZE, MISC:ptr, MISC:ptr:
    - nppiFusedAbsDiff_Threshold_GTVal (1)
  MISC:NppDataType, MISC:NppiChannels, MISC:ptr, MISC:i32, MISC:ptr, MISC:i32, SIZE, MISC:ptr, MISC:ptr:
    - nppiFusedAbsDiff_Threshold_GTVal_I (1)
  MISC:NppiHOGConfig, MISC:i32, MISC:i32:
    - nppiHistogramOfGradientsBorderGetDescriptorsSize (1)
  MISC:NppiHOGConfig, MISC:ptr, MISC:i32, SIZE, MISC:i32:
    - nppiHistogramOfGradientsBorderGetBufferSize (1)
  MISC:c_uint, SCRATCH_BUF:
    - nppiCompressedMarkerLabelsUFGetInfoListSize_ (1)
  MISC:i32, MISC:i32:
    - nppiCompressMarkerLabelsGetBufferSize_ (1)
  MISC:i32, MISC:i32, MISC:ptr, MISC:Npp32f:
    - nppiGetFilterGaussPyramidLayerDownBorderDstROI (1)
  MISC:i32, MISC:i32, MISC:ptr, MISC:ptr, MISC:Npp32f:
    - nppiGetFilterGaussPyramidLayerUpBorderDstROI (1)
  MISC:i32, MISC:i32, MISC:ptr, MISC:ptr, MISC:ptr, MISC:c_uint, INTERP:
    - nppiResizeBatch_ (11)
  MISC:i32, MISC:i32, i32:step:
    - nppiMalloc_ (30)
  MISC:ptr:
    - nppiFree (1)
  MISC:ptr, DST+STEP, SIZE:
    - nppiSet_ (12)
  MISC:ptr, MISC:c_uint:
    - nppiWarpAffineBatchInit (1)
    - nppiWarpPerspectiveBatchInit (1)
  MISC:ptr, MISC:i32, DST+STEP, SIZE:
    - nppiCopy_ (10)
  MISC:ptr, MISC:i32, MISC:Npp16sc, DST+STEP, SIZE, CONST_SCALAR:
    - nppiAddC_ (1)
    - nppiDivC_ (1)
    - nppiMulC_ (1)
    - nppiSubC_ (1)
  MISC:ptr, MISC:i32, MISC:Npp32f, DST+STEP, SIZE:
    - nppiAddC_ (1)
    - nppiDivC_ (1)
    - nppiMulC_ (1)
    - nppiSubC_ (1)
  MISC:ptr, MISC:i32, MISC:Npp32fc, DST+STEP, SIZE:
    - nppiAddC_ (1)
    - nppiDivC_ (1)
    - nppiMulC_ (1)
    - nppiSubC_ (1)
  MISC:ptr, MISC:i32, MISC:Npp32sc, DST+STEP, SIZE, CONST_SCALAR:
    - nppiAddC_ (1)
    - nppiDivC_ (1)
    - nppiMulC_ (1)
    - nppiSubC_ (1)
  MISC:ptr, MISC:i32, MISC:ptr, DST+STEP, SIZE:
    - nppiAddC_ (3)
    - nppiDivC_ (3)
    - nppiMulC_ (3)
    - nppiSubC_ (3)
  MISC:ptr, MISC:i32, MISC:ptr, DST+STEP, SIZE, CONST_SCALAR:
    - nppiAddC_ (4)
    - nppiDivC_ (4)
    - nppiMulC_ (4)
    - nppiSubC_ (4)
  MISC:ptr, MISC:i32, MISC:ptr, MISC:i32, DST+STEP, SIZE:
    - nppiAbsDiff_ (1)
    - nppiAdd_ (7)
    - nppiDiv_ (7)
    - nppiMul_ (7)
    - nppiSub_ (7)
  MISC:ptr, MISC:i32, MISC:ptr, MISC:i32, DST+STEP, SIZE, CONST_SCALAR:
    - nppiAdd_ (6)
    - nppiDiv_ (6)
    - nppiMul_ (6)
    - nppiSub_ (6)
  MISC:ptr, MISC:i32, MISC:ptr, MISC:i32, MISC:ptr, MISC:i32, SIZE:
    - nppiAddProduct_ (1)
  MISC:ptr, MISC:i32, MISC:ptr, MISC:i32, SIZE, ptr:dst, ptr:dst:
    - nppiAverageError_ (12)
    - nppiAverageRelativeError_ (12)
    - nppiMaximumError_ (12)
    - nppiMaximumRelativeError_ (12)
  MISC:ptr, MISC:i32, SIZE:
    - nppiAbs_ (3)
    - nppiGammaFwd_ (1)
    - nppiGammaInv_ (1)
    - nppiLn_ (2)
    - nppiSqr_ (3)
    - nppiSqrt_ (3)
  MISC:ptr, MISC:i32, SIZE, MISC:ptr:
    - nppiColorTwist (8)
    - nppiColorTwist_ (1)
  MISC:ptr, MISC:i32, SIZE, MISC:ptr, ptr:src:
    - nppiColorTwist (1)
  MISC:ptr, MISC:i32, ptr:src, DST+STEP, SIZE:
    - nppiAddC_ (2)
    - nppiAddDeviceC_ (3)
    - nppiDivC_ (2)
    - nppiDivDeviceC_ (3)
    - nppiMulC_ (2)
    - nppiMulDeviceC_ (3)
    - nppiSubC_ (2)
    - nppiSubDeviceC_ (3)
  MISC:ptr, MISC:ptr, MISC:i32, SIZE:
    - nppiAddC_ (3)
    - nppiDivC_ (3)
    - nppiMulC_ (3)
    - nppiSubC_ (3)
    - nppiYCbCr (8)
    - nppiYCbCrToBGRBatch_ (4)
    - nppiYCbCrToRGBBatch_ (4)
    - nppiYUV (8)
    - nppiYUVToBGRBatch_ (4)
    - nppiYUVToRGBBatch_ (4)
  MISC:ptr, MISC:ptr, MISC:i32, SIZE, CONST_SCALAR:
    - nppiAddC_ (4)
    - nppiDivC_ (4)
    - nppiMulC_ (4)
    - nppiSubC_ (4)
  MISC:ptr, MISC:ptr, MISC:i32, SIZE, MISC:NppiNorm:
    - nppiLabelMarkersUFBatch_ (6)
  MISC:ptr, MISC:ptr, MISC:i32, SIZE, ptr:dst, MISC:ptr:
    - nppiMSEBatch_ (4)
    - nppiPSNRBatch_ (4)
    - nppiSSIMBatch_ (2)
    - nppiWMSSSIMBatch_ (4)
  MISC:ptr, MISC:ptr, MISC:ptr, MISC:Npp32s, MISC:ptr, MISC:ptr, ptr:dst, MISC:Npp32s, ptr:dst, ptr:dst, ptr:dst, ptr:dst, ptr:dst, MISC:Npp32u, MISC:Npp32u, MISC:Npp32u, MISC:Npp32u, MISC:ptr, MISC:ptr, MISC:Npp32u, SIZE:
    - nppiCompressedMarkerLabelsUFContoursGenerateGeometryLists_C (1)
  MISC:ptr, MISC:ptr, MISC:ptr, MISC:i32, SIZE, MISC:i32:
    - nppiCompressMarkerLabelsUFBatch_ (2)
  MISC:ptr, SIZE, MISC:i32, RECT, DST+STEP, RECT, MISC:f64, MISC:f64, MISC:f64, MISC:f64, INTERP:
    - nppiResizeSqrPixel_ (10)
  MISC:ptr, SIZE, MISC:i32, RECT, DST+STEP, RECT, MISC:ptr, INTERP:
    - nppiWarpAffineBack_ (8)
    - nppiWarpAffine_ (11)
    - nppiWarpPerspectiveBack_ (8)
    - nppiWarpPerspective_ (11)
  MISC:ptr, SIZE, MISC:i32, RECT, MISC:ptr, DST+STEP, RECT, MISC:ptr, INTERP:
    - nppiWarpAffineQuad_ (8)
    - nppiWarpPerspectiveQuad_ (8)
  MISC:ptr, SIZE, MISC:i32, RECT, MISC:ptr, MISC:i32, RECT, MISC:ptr, INTERP:
    - nppiWarpAffine_ (2)
  MISC:ptr, SIZE, MISC:i32, RECT, ptr:src, MISC:i32, ptr:src, MISC:i32, DST+STEP, SIZE, INTERP:
    - nppiRemap_ (10)
  RECT, MISC:ptr, MISC:f64, MISC:f64, MISC:f64:
    - nppiGetRotateBound (1)
    - nppiGetRotateQuad (1)
  RECT, MISC:ptr, MISC:f64, MISC:f64, MISC:f64, MISC:f64, INTERP:
    - nppiGetResizeRect (1)
  RECT, MISC:ptr, MISC:ptr:
    - nppiGetAffineBound (1)
    - nppiGetAffineQuad (1)
    - nppiGetAffineTransform (1)
    - nppiGetPerspectiveBound (1)
    - nppiGetPerspectiveQuad (1)
    - nppiGetPerspectiveTransform (1)
  RECT, RECT, MISC:ptr:
    - nppiGetResizeTiledSourceOffset (1)
  SIZE, MISC:NppPointPolar, MISC:i32, MISC:i32:
    - nppiFilterHoughLineGetBufferSize (1)
  SIZE, MISC:NppiAxis, MISC:ptr, MISC:i32:
    - nppiMirrorBatch_ (8)
  SIZE, MISC:i32:
    - nppiFilterCannyBorderGetBufferSize (1)
    - nppiFilterHarrisCornersBorderGetBufferSize (1)
    - nppiFloodFillGetBufferSize (1)
    - nppiLabelMarkersUFGetBufferSize_ (1)
    - nppiMorphGetBufferSize_ (8)
  SIZE, MISC:i32, MISC:i32:
    - nppiFilterBoxBorderAdvancedGetDeviceBufferSize (1)
    - nppiFilterBoxBorderAdvancedGetDeviceBufferSize_ (1)
  SIZE, MISC:i32, MISC:ptr:
    - nppiHistogramEvenGetBufferSize_ (12)
    - nppiHistogramRangeGetBufferSize_ (16)
  SIZE, MISC:ptr:
    - nppiAverageErrorGetBufferHostSize_ (44)
    - nppiAverageRelativeErrorGetBufferHostSize_ (44)
    - nppiCountInRangeGetBufferHostSize_ (6)
    - nppiDistanceTransformPBAGetAntialiasingBufferSize (1)
    - nppiDistanceTransformPBAGetBufferSize (1)
    - nppiDotProdGetBufferHostSize_ (28)
    - nppiFullNormLevelGetBufferHostSize_ (24)
    - nppiMSEBatchGetBufferHostSize_ (2)
    - nppiMSEGetBufferHostSize_ (2)
    - nppiMSSSIMGetBufferHostSize_ (1)
    - nppiMaxGetBufferHostSize_ (16)
    - nppiMaxIndxGetBufferHostSize_ (16)
    - nppiMaximumErrorGetBufferHostSize_ (44)
    - nppiMaximumRelativeErrorGetBufferHostSize_ (44)
    - nppiMeanGetBufferHostSize_ (24)
    - nppiMeanStdDevGetBufferHostSize_ (16)
    - nppiMinGetBufferHostSize_ (16)
    - nppiMinIndxGetBufferHostSize_ (16)
    - nppiMinMaxGetBufferHostSize_ (16)
    - nppiMinMaxIndxGetBufferHostSize_ (16)
    - nppiNormDiffInfGetBufferHostSize_ (24)
    - nppiNormDiffL (48)
    - nppiNormInfGetBufferHostSize_ (25)
    - nppiNormL (48)
    - nppiNormRelInfGetBufferHostSize_ (24)
    - nppiNormRelL (48)
    - nppiPSNRBatchGetBufferHostSize_ (2)
    - nppiPSNRGetBufferHostSize_ (2)
    - nppiQualityIndexGetBufferHostSize_ (9)
    - nppiSSIMBatchGetBufferHostSize_ (2)
    - nppiSSIMGetBufferHostSize_ (2)
    - nppiSameNormLevelGetBufferHostSize_ (24)
    - nppiSegmentWatershedGetBufferSize_ (2)
    - nppiSignedDistanceTransformPBAGet (1)
    - nppiSignedDistanceTransformPBAGetAntialiasingBufferSize (1)
    - nppiSignedDistanceTransformPBAGetBufferSize (1)
    - nppiSumGetBufferHostSize_ (18)
    - nppiValidNormLevelGetBufferHostSize_ (24)
    - nppiWMSSSIMBatchGetBufferHostSize_ (2)
    - nppiWMSSSIMGetBufferHostSize_ (2)
  SIZE, RECT, RECT, INTERP, MISC:ptr, MISC:c_uint:
    - nppiWarpAffineBatch_ (11)
    - nppiWarpPerspectiveBatch_ (11)
  SIZE, RECT, SIZE, RECT, INTERP, MISC:ptr, MISC:c_uint:
    - nppiResizeBatch_ (8)
  SIZE, SIZE, MISC:i32, INTERP:
    - nppiResizeAdvancedGetBufferHostSize_ (1)
  SIZE, SIZE, MISC:i32, MISC:i32, MISC:ptr:
    - nppiCrossCorrFull_NormLevel_GetAdvancedScratchBufferSize (1)
    - nppiCrossCorrSame_NormLevel_GetAdvancedScratchBufferSize (1)
    - nppiCrossCorrValid_NormLevel_GetAdvancedScratchBufferSize (1)
  SIZE, SIZE, ptr:dst:
    - nppiFilterMedianGetBufferSize_ (16)
  SIZE, SIZE, ptr:dst, MISC:NppiBorderType:
    - nppiFilterMedianBorderGetBufferSize_ (16)
  SRC+STEP, CONST_SCALAR, DST+STEP, SIZE, MISC:ptr, MISC:i32:
    - nppiLUTPaletteSwap_ (2)
  SRC+STEP, DST+STEP, SIZE:
    - nppiAbs_ (11)
    - nppiBGRToCbYCr (3)
    - nppiBGRToHLS_ (7)
    - nppiBGRToLab_ (1)
    - nppiBGRToYCbCr (19)
    - nppiBGRToYCbCr_ (3)
    - nppiBGRToYCrCb (4)
    - nppiBGRToYUV (1)
    - nppiBGRToYUV_ (5)
    - nppiCMYKOrYCCKToBGR_JPEG_ (2)
    - nppiCMYKOrYCCKToRGB_JPEG_ (2)
    - nppiCbYCr (7)
    - nppiConvert_ (70)
    - nppiCopy_ (75)
    - nppiDilate (6)
    - nppiDup_ (15)
    - nppiErode (6)
    - nppiExp_ (2)
    - nppiGammaFwd_ (3)
    - nppiGammaInv_ (3)
    - nppiHLSToBGR_ (6)
    - nppiHLSToRGB_ (2)
    - nppiHSVToRGB_ (2)
    - nppiLUVToRGB_ (2)
    - nppiLabToBGR_ (1)
    - nppiLn_ (4)
    - nppiMagnitudeSqr_ (1)
    - nppiMagnitude_ (1)
    - nppiNV (9)
    - nppiNot_ (4)
    - nppiRGBToCbYCr (2)
    - nppiRGBToGray_ (8)
    - nppiRGBToHLS_ (2)
    - nppiRGBToHSV_ (2)
    - nppiRGBToLUV_ (2)
    - nppiRGBToXYZ_ (2)
    - nppiRGBToYCC_ (2)
    - nppiRGBToYCbCr (14)
    - nppiRGBToYCbCr_ (5)
    - nppiRGBToYCrCb (3)
    - nppiRGBToYUV (5)
    - nppiRGBToYUV_ (5)
    - nppiScale_ (12)
    - nppiSqr_ (7)
    - nppiSqrt_ (7)
    - nppiTranspose_ (15)
    - nppiXYZToRGB_ (2)
    - nppiYCCKToCMYK_JPEG_ (1)
    - nppiYCCToRGB_ (2)
    - nppiYCbCr (42)
    - nppiYCbCrToBGR_ (2)
    - nppiYCbCrToRGB_ (4)
    - nppiYCrCb (8)
    - nppiYUV (10)
    - nppiYUVToBGR_ (4)
    - nppiYUVToRGB_ (4)
  SRC+STEP, DST+STEP, SIZE, CONST_SCALAR:
    - nppiExp_ (6)
    - nppiLn_ (6)
    - nppiSqr_ (12)
    - nppiSqrt_ (9)
  SRC+STEP, DST+STEP, SIZE, MISC:Npp16s:
    - nppiThreshold_GT_ (1)
    - nppiThreshold_LT_ (1)
  SRC+STEP, DST+STEP, SIZE, MISC:Npp16s, MISC:Npp16s:
    - nppiThreshold_GTVal_ (1)
    - nppiThreshold_LTVal_ (1)
  SRC+STEP, DST+STEP, SIZE, MISC:Npp16s, MISC:Npp16s, MISC:Npp16s, MISC:Npp16s:
    - nppiThreshold_LTValGTVal_ (1)
  SRC+STEP, DST+STEP, SIZE, MISC:Npp16s, MISC:Npp16s, MISC:NppCmpOp:
    - nppiThreshold_Val_ (1)
  SRC+STEP, DST+STEP, SIZE, MISC:Npp16s, MISC:NppCmpOp:
    - nppiThreshold_ (1)
  SRC+STEP, DST+STEP, SIZE, MISC:Npp16u:
    - nppiThreshold_GT_ (1)
    - nppiThreshold_LT_ (1)
  SRC+STEP, DST+STEP, SIZE, MISC:Npp16u, MISC:Npp16u:
    - nppiThreshold_GTVal_ (1)
    - nppiThreshold_LTVal_ (1)
  SRC+STEP, DST+STEP, SIZE, MISC:Npp16u, MISC:Npp16u, MISC:Npp16u, MISC:Npp16u:
    - nppiThreshold_LTValGTVal_ (1)
  SRC+STEP, DST+STEP, SIZE, MISC:Npp16u, MISC:Npp16u, MISC:NppCmpOp:
    - nppiThreshold_Val_ (1)
  SRC+STEP, DST+STEP, SIZE, MISC:Npp16u, MISC:NppCmpOp:
    - nppiThreshold_ (1)
  SRC+STEP, DST+STEP, SIZE, MISC:Npp32f:
    - nppiIntegral_ (1)
    - nppiThreshold_GT_ (1)
    - nppiThreshold_LT_ (1)
  SRC+STEP, DST+STEP, SIZE, MISC:Npp32f, MISC:Npp32f:
    - nppiCopySubpix_ (20)
    - nppiScale_ (8)
    - nppiThreshold_GTVal_ (1)
    - nppiThreshold_LTVal_ (1)
  SRC+STEP, DST+STEP, SIZE, MISC:Npp32f, MISC:Npp32f, MISC:Npp32f, MISC:Npp32f:
    - nppiThreshold_LTValGTVal_ (1)
  SRC+STEP, DST+STEP, SIZE, MISC:Npp32f, MISC:Npp32f, MISC:NppCmpOp:
    - nppiThreshold_Val_ (1)
  SRC+STEP, DST+STEP, SIZE, MISC:Npp32f, MISC:NppCmpOp:
    - nppiThreshold_ (1)
  SRC+STEP, DST+STEP, SIZE, MISC:Npp32s:
    - nppiIntegral_ (1)
  SRC+STEP, DST+STEP, SIZE, MISC:Npp8u:
    - nppiCbYCr (2)
    - nppiThreshold_GT_ (1)
    - nppiThreshold_LT_ (1)
    - nppiYCbCr (5)
    - nppiYCbCrToBGR_ (2)
    - nppiYCbCrToRGB_ (1)
    - nppiYCrCb (1)
  SRC+STEP, DST+STEP, SIZE, MISC:Npp8u, MISC:Npp8u:
    - nppiThreshold_GTVal_ (1)
    - nppiThreshold_LTVal_ (1)
  SRC+STEP, DST+STEP, SIZE, MISC:Npp8u, MISC:Npp8u, MISC:Npp8u, MISC:Npp8u:
    - nppiThreshold_LTValGTVal_ (1)
  SRC+STEP, DST+STEP, SIZE, MISC:Npp8u, MISC:Npp8u, MISC:NppCmpOp:
    - nppiThreshold_Val_ (1)
  SRC+STEP, DST+STEP, SIZE, MISC:Npp8u, MISC:NppCmpOp:
    - nppiThreshold_ (1)
  SRC+STEP, DST+STEP, SIZE, MISC:NppHintAlgorithm:
    - nppiScale_ (12)
  SRC+STEP, DST+STEP, SIZE, MISC:NppRoundMode:
    - nppiConvert_ (20)
  SRC+STEP, DST+STEP, SIZE, MISC:NppRoundMode, CONST_SCALAR:
    - nppiConvert_ (17)
  SRC+STEP, DST+STEP, SIZE, MISC:NppiAxis:
    - nppiMirror_ (20)
  SRC+STEP, DST+STEP, SIZE, MISC:NppiNorm:
    - nppiGradientColorToGray_ (4)
  SRC+STEP, DST+STEP, SIZE, MISC:NppiNorm, ptr:dst:
    - nppiLabelMarkersUF_ (3)
  SRC+STEP, DST+STEP, SIZE, MISC:i32:
    - nppiSwapChannels_ (20)
  SRC+STEP, DST+STEP, SIZE, MISC:i32, MISC:Npp16s:
    - nppiSwapChannels_ (1)
  SRC+STEP, DST+STEP, SIZE, MISC:i32, MISC:Npp16u:
    - nppiSwapChannels_ (1)
  SRC+STEP, DST+STEP, SIZE, MISC:i32, MISC:Npp32f:
    - nppiSwapChannels_ (1)
  SRC+STEP, DST+STEP, SIZE, MISC:i32, MISC:Npp32s:
    - nppiSwapChannels_ (1)
  SRC+STEP, DST+STEP, SIZE, MISC:i32, MISC:Npp8u:
    - nppiSwapChannels_ (1)
  SRC+STEP, DST+STEP, SIZE, MISC:ptr:
    - nppiColorTwist (26)
    - nppiColorTwist_ (6)
    - nppiNV (2)
    - nppiRGBToNV (4)
    - nppiRGBToYUV (10)
    - nppiYUV (12)
  SRC+STEP, DST+STEP, SIZE, MISC:ptr, MISC:Npp16u:
    - nppiYUV (2)
  SRC+STEP, DST+STEP, SIZE, MISC:ptr, MISC:Npp8u:
    - nppiYUV (2)
  SRC+STEP, DST+STEP, SIZE, MISC:ptr, MISC:i32:
    - nppiLUTPalette_ (6)
  SRC+STEP, DST+STEP, SIZE, MISC:ptr, MISC:ptr, MISC:i32:
    - nppiLUT_ (12)
    - nppiLUT_Cubic_ (12)
    - nppiLUT_Linear_ (12)
  SRC+STEP, DST+STEP, SIZE, MISC:ptr, ptr:src:
    - nppiColorTwist (2)
    - nppiColorTwist_ (1)
  SRC+STEP, DST+STEP, SIZE, ptr:dst, MISC:ptr, MISC:i32:
    - nppiLUT_Trilinear_ (2)
  SRC+STEP, DST+STEP, SIZE, ptr:src:
    - nppiColorToGray_ (12)
    - nppiThreshold_GT_ (8)
    - nppiThreshold_LT_ (8)
  SRC+STEP, DST+STEP, SIZE, ptr:src, MISC:NppCmpOp:
    - nppiThreshold_ (8)
  SRC+STEP, DST+STEP, SIZE, ptr:src, MISC:i32:
    - nppiCopy_ (20)
    - nppiLUTPalette_ (7)
  SRC+STEP, DST+STEP, SIZE, ptr:src, SIZE, POINT:
    - nppiDilate_ (6)
    - nppiErode_ (6)
    - nppiFilter (33)
  SRC+STEP, DST+STEP, SIZE, ptr:src, ptr:src:
    - nppiThreshold_GTVal_ (8)
    - nppiThreshold_LTVal_ (8)
  SRC+STEP, DST+STEP, SIZE, ptr:src, ptr:src, MISC:NppCmpOp:
    - nppiThreshold_Val_ (8)
  SRC+STEP, DST+STEP, SIZE, ptr:src, ptr:src, MISC:i32:
    - nppiLUT_ (4)
    - nppiLUT_Cubic_ (4)
    - nppiLUT_Linear_ (4)
  SRC+STEP, DST+STEP, SIZE, ptr:src, ptr:src, ptr:src, ptr:src:
    - nppiThreshold_LTValGTVal_ (8)
  SRC+STEP, DST+STEP, ptr:dst, MISC:i32, SIZE, MISC:Npp32f, MISC:Npp64f:
    - nppiSqrIntegral_ (1)
  SRC+STEP, DST+STEP, ptr:dst, MISC:i32, SIZE, MISC:Npp32s, MISC:Npp32s:
    - nppiSqrIntegral_ (1)
  SRC+STEP, DST+STEP, ptr:dst, MISC:i32, SIZE, MISC:Npp32s, MISC:Npp64f:
    - nppiSqrIntegral_ (1)
  SRC+STEP, MISC:Npp16s, DST+STEP, SIZE, MISC:NppCmpOp:
    - nppiCompareC_ (1)
  SRC+STEP, MISC:Npp16s, MISC:Npp16s, ptr:dst, MISC:i32, ptr:dst, MISC:i32, ptr:dst, MISC:i32, ptr:dst, MISC:i32, SIZE, ptr:dst:
    - nppiDistanceTransformAbsPBA_ (2)
    - nppiDistanceTransformPBA_ (2)
  SRC+STEP, MISC:Npp16s, MISC:Npp16s, ptr:dst, MISC:i32, ptr:dst, MISC:i32, ptr:dst, MISC:i32, ptr:dst, MISC:i32, SIZE, ptr:dst, ptr:dst:
    - nppiDistanceTransformAbsPBA_ (1)
    - nppiDistanceTransformPBA_ (1)
  SRC+STEP, MISC:Npp16u, DST+STEP, SIZE, MISC:NppCmpOp:
    - nppiCompareC_ (1)
  SRC+STEP, MISC:Npp16u, MISC:Npp16u, ptr:dst, MISC:i32, ptr:dst, MISC:i32, ptr:dst, MISC:i32, ptr:dst, MISC:i32, SIZE, ptr:dst:
    - nppiDistanceTransformAbsPBA_ (2)
    - nppiDistanceTransformPBA_ (2)
  SRC+STEP, MISC:Npp16u, MISC:Npp16u, ptr:dst, MISC:i32, ptr:dst, MISC:i32, ptr:dst, MISC:i32, ptr:dst, MISC:i32, SIZE, ptr:dst, ptr:dst:
    - nppiDistanceTransformAbsPBA_ (1)
    - nppiDistanceTransformPBA_ (1)
  SRC+STEP, MISC:Npp32f, DST+STEP, SIZE, MISC:Npp32f:
    - nppiCompareEqualEpsC_ (1)
  SRC+STEP, MISC:Npp32f, DST+STEP, SIZE, MISC:NppCmpOp:
    - nppiCompareC_ (1)
  SRC+STEP, MISC:Npp32f, MISC:Npp32f, MISC:Npp32f, ptr:dst, MISC:i32, ptr:dst, MISC:i32, ptr:dst, MISC:i32, ptr:dst, MISC:i32, SIZE, ptr:dst:
    - nppiSignedDistanceTransformAbsPBA_ (1)
    - nppiSignedDistanceTransformPBA_ (1)
  SRC+STEP, MISC:Npp32f, MISC:Npp32f, ptr:dst, MISC:i32, ptr:dst, MISC:i32, ptr:dst, MISC:i32, ptr:dst, MISC:i32, SIZE, ptr:dst, ptr:dst:
    - nppiDistanceTransformAbsPBA_ (1)
    - nppiDistanceTransformPBA_ (1)
  SRC+STEP, MISC:Npp32f, MISC:Npp64f, MISC:Npp64f, ptr:dst, MISC:i32, ptr:dst, MISC:i32, ptr:dst, MISC:i32, ptr:dst, MISC:i32, SIZE, ptr:dst, ptr:dst:
    - nppiSignedDistanceTransformAbsPBA_ (1)
    - nppiSignedDistanceTransformPBA_ (1)
  SRC+STEP, MISC:Npp64f, MISC:Npp64f, MISC:Npp64f, ptr:dst, MISC:i32, ptr:dst, MISC:i32, ptr:dst, MISC:i32, ptr:dst, MISC:i32, SIZE, ptr:dst, ptr:dst:
    - nppiSignedDistanceTransformAbsPBA_ (1)
    - nppiSignedDistanceTransformPBA_ (1)
  SRC+STEP, MISC:Npp64f, MISC:Npp64f, ptr:dst, MISC:i32, ptr:dst, MISC:i32, ptr:dst, MISC:i32, ptr:dst, MISC:i32, SIZE, ptr:dst, ptr:dst:
    - nppiDistanceTransformAbsPBA_ (1)
    - nppiDistanceTransformPBA_ (1)
  SRC+STEP, MISC:Npp8s, MISC:Npp8s, ptr:dst, MISC:i32, ptr:dst, MISC:i32, ptr:dst, MISC:i32, ptr:dst, MISC:i32, SIZE, ptr:dst:
    - nppiDistanceTransformAbsPBA_ (2)
    - nppiDistanceTransformPBA_ (2)
  SRC+STEP, MISC:Npp8s, MISC:Npp8s, ptr:dst, MISC:i32, ptr:dst, MISC:i32, ptr:dst, MISC:i32, ptr:dst, MISC:i32, SIZE, ptr:dst, ptr:dst:
    - nppiDistanceTransformAbsPBA_ (1)
    - nppiDistanceTransformPBA_ (1)
  SRC+STEP, MISC:Npp8u, DST+STEP, SIZE, MISC:NppCmpOp:
    - nppiCompareC_ (1)
  SRC+STEP, MISC:Npp8u, MISC:Npp8u, ptr:dst, MISC:i32, ptr:dst, MISC:i32, ptr:dst, MISC:i32, ptr:dst, MISC:i32, SIZE, ptr:dst:
    - nppiDistanceTransformAbsPBA_ (2)
    - nppiDistanceTransformPBA_ (2)
  SRC+STEP, MISC:Npp8u, MISC:Npp8u, ptr:dst, MISC:i32, ptr:dst, MISC:i32, ptr:dst, MISC:i32, ptr:dst, MISC:i32, SIZE, ptr:dst, ptr:dst:
    - nppiDistanceTransformAbsPBA_ (1)
    - nppiDistanceTransformPBA_ (1)
  SRC+STEP, MISC:ptr, MISC:i32, SIZE:
    - nppiAdd_ (7)
    - nppiCopy_ (10)
    - nppiDiv_ (7)
    - nppiMul_ (7)
    - nppiSub_ (7)
  SRC+STEP, MISC:ptr, MISC:i32, SIZE, CONST_SCALAR:
    - nppiAdd_ (6)
    - nppiDiv_ (6)
    - nppiMul_ (6)
    - nppiSub_ (6)
  SRC+STEP, SIZE, DST+STEP, SIZE, MISC:i32, MISC:i32:
    - nppiCopyReplicateBorder_ (20)
    - nppiCopyWrapBorder_ (20)
  SRC+STEP, SIZE, DST+STEP, SIZE, MISC:i32, MISC:i32, MISC:Npp16s:
    - nppiCopyConstBorder_ (1)
  SRC+STEP, SIZE, DST+STEP, SIZE, MISC:i32, MISC:i32, MISC:Npp16u:
    - nppiCopyConstBorder_ (1)
  SRC+STEP, SIZE, DST+STEP, SIZE, MISC:i32, MISC:i32, MISC:Npp32f:
    - nppiCopyConstBorder_ (1)
  SRC+STEP, SIZE, DST+STEP, SIZE, MISC:i32, MISC:i32, MISC:Npp32s:
    - nppiCopyConstBorder_ (1)
  SRC+STEP, SIZE, DST+STEP, SIZE, MISC:i32, MISC:i32, MISC:Npp8u:
    - nppiCopyConstBorder_ (1)
  SRC+STEP, SIZE, DST+STEP, SIZE, MISC:i32, MISC:i32, ptr:src:
    - nppiCopyConstBorder_ (15)
  SRC+STEP, SIZE, MISC:NppPointPolar, MISC:i32, MISC:ptr, MISC:i32, MISC:i32, ptr:dst:
    - nppiFilterHoughLine_ (1)
  SRC+STEP, SIZE, MISC:NppPointPolar, MISC:i32, MISC:ptr, MISC:ptr, MISC:i32, MISC:i32, ptr:dst:
    - nppiFilterHoughLineRegion_ (1)
  SRC+STEP, SIZE, MISC:NppiPoint32f, MISC:Npp32f, MISC:Npp32f, MISC:ptr, MISC:Npp16u:
    - nppiEllipticalRadialProfile_ (1)
  SRC+STEP, SIZE, MISC:NppiPoint32f, MISC:ptr, MISC:Npp16u:
    - nppiCircularRadialProfile_ (1)
  SRC+STEP, SIZE, MISC:i32, MISC:Npp32f, MISC:Npp32f, ptr:dst:
    - nppiCountInRange_ (1)
  SRC+STEP, SIZE, MISC:i32, MISC:Npp8u, MISC:Npp8u, ptr:dst:
    - nppiCountInRange_ (1)
  SRC+STEP, SIZE, MISC:i32, ptr:dst, ptr:dst, MISC:ptr, MISC:ptr, ptr:dst:
    - nppiMinMaxIndx_ (4)
  SRC+STEP, SIZE, MISC:i32, ptr:dst, ptr:dst, ptr:dst:
    - nppiCountInRange_ (4)
    - nppiMean_StdDev_ (4)
  SRC+STEP, SIZE, MISC:ptr, MISC:i32, ptr:dst, ptr:dst, ptr:dst:
    - nppiHistogramEven_ (9)
  SRC+STEP, SIZE, MISC:ptr, MISC:ptr, MISC:i32, ptr:dst:
    - nppiHistogramRange_ (12)
  SRC+STEP, SIZE, POINT, DST+STEP, SIZE, MISC:NppiBorderType:
    - nppiDilate (6)
    - nppiErode (6)
  SRC+STEP, SIZE, POINT, DST+STEP, SIZE, MISC:NppiDifferentialKernel, MISC:NppiMaskSize, MISC:Npp16s, MISC:Npp16s, MISC:NppiNorm, MISC:NppiBorderType, ptr:dst:
    - nppiFilterCannyBorder_ (1)
  SRC+STEP, SIZE, POINT, DST+STEP, SIZE, MISC:NppiDifferentialKernel, MISC:NppiMaskSize, MISC:NppiMaskSize, MISC:Npp32f, MISC:Npp32f, MISC:NppiBorderType, ptr:dst:
    - nppiFilterHarrisCornersBorder_ (1)
  SRC+STEP, SIZE, POINT, DST+STEP, SIZE, ptr:src, MISC:Npp32s, MISC:Npp32s, MISC:NppiBorderType:
    - nppiFilterColumnBorder (12)
    - nppiFilterRowBorder (12)
  SRC+STEP, SIZE, POINT, DST+STEP, SIZE, ptr:src, SIZE, POINT, MISC:NppiBorderType:
    - nppiDilateBorder_ (6)
    - nppiErodeBorder_ (6)
    - nppiFilterBorder (33)
  SRC+STEP, SIZE, POINT, DST+STEP, SIZE, ptr:src, SIZE, POINT, ptr:dst, MISC:NppiBorderType:
    - nppiMorphBlackHatBorder_ (8)
    - nppiMorphCloseBorder_ (8)
    - nppiMorphGradientBorder_ (8)
    - nppiMorphOpenBorder_ (8)
    - nppiMorphTopHatBorder_ (8)
  SRC+STEP, SIZE, POINT, MISC:ptr, MISC:i32, ptr:dst, SIZE, MISC:NppiHOGConfig, ptr:dst, MISC:NppiBorderType:
    - nppiHistogramOfGradientsBorder_ (8)
  SRC+STEP, SIZE, POINT, ptr:dst, MISC:i32, ptr:dst, MISC:i32, ptr:dst, MISC:i32, ptr:dst, MISC:i32, SIZE, MISC:NppiMaskSize, MISC:NppiNorm, MISC:NppiBorderType:
    - nppiGradientVectorPrewittBorder_ (8)
    - nppiGradientVectorScharrBorder_ (8)
    - nppiGradientVectorSobelBorder_ (8)
  SRC+STEP, SIZE, RECT, DST+STEP, MISC:NppiBayerGridPosition, MISC:NppiInterpolationMode:
    - nppiCFAToRGB_ (3)
  SRC+STEP, SIZE, RECT, DST+STEP, MISC:NppiBayerGridPosition, MISC:NppiInterpolationMode, MISC:Npp16u:
    - nppiCFAToRGBA_ (1)
  SRC+STEP, SIZE, RECT, DST+STEP, MISC:NppiBayerGridPosition, MISC:NppiInterpolationMode, MISC:Npp32u:
    - nppiCFAToRGBA_ (1)
  SRC+STEP, SIZE, RECT, DST+STEP, MISC:NppiBayerGridPosition, MISC:NppiInterpolationMode, MISC:Npp8u:
    - nppiCFAToRGBA_ (1)
  SRC+STEP, SIZE, RECT, DST+STEP, SIZE, RECT, INTERP:
    - nppiResize_ (27)
  SRC+STEP, SIZE, ptr:dst, MISC:i32, MISC:Npp32s, MISC:Npp32s, ptr:dst:
    - nppiHistogramEven_ (3)
  SRC+STEP, SIZE, ptr:dst, MISC:ptr:
    - nppiSum_ (2)
  SRC+STEP, SIZE, ptr:dst, ptr:dst:
    - nppiMax_ (16)
    - nppiMean_ (16)
    - nppiMin_ (16)
    - nppiNorm_Inf_ (17)
    - nppiNorm_L (32)
    - nppiSum_ (16)
  SRC+STEP, SIZE, ptr:dst, ptr:dst, MISC:i32, MISC:i32:
    - nppiMaxIndx_ (16)
    - nppiMinIndx_ (16)
  SRC+STEP, SIZE, ptr:dst, ptr:dst, MISC:ptr, MISC:ptr, ptr:dst:
    - nppiMinMaxIndx_ (4)
  SRC+STEP, SIZE, ptr:dst, ptr:dst, ptr:dst:
    - nppiMean_StdDev_ (4)
    - nppiMinMax_ (16)
  SRC+STEP, SIZE, ptr:dst, ptr:src, MISC:i32, ptr:dst:
    - nppiHistogramRange_ (4)
  SRC+STEP, SIZE, ptr:src, MISC:i32, SIZE, DST+STEP:
    - nppiCrossCorrFull_Norm_ (20)
    - nppiCrossCorrSame_Norm_ (20)
    - nppiCrossCorrValid_ (5)
    - nppiCrossCorrValid_Norm_ (20)
    - nppiSqrDistanceFull_Norm_ (16)
    - nppiSqrDistanceSame_Norm_ (16)
    - nppiSqrDistanceValid_Norm_ (16)
  SRC+STEP, SIZE, ptr:src, MISC:i32, SIZE, DST+STEP, CONST_SCALAR:
    - nppiCrossCorrFull_Norm_ (4)
    - nppiCrossCorrSame_Norm_ (4)
    - nppiCrossCorrValid_Norm_ (4)
    - nppiSqrDistanceFull_Norm_ (4)
    - nppiSqrDistanceSame_Norm_ (4)
    - nppiSqrDistanceValid_Norm_ (4)
  SRC+STEP, SIZE, ptr:src, MISC:i32, SIZE, DST+STEP, CONST_SCALAR, ptr:dst:
    - nppiCrossCorrFull_NormLevel_ (4)
    - nppiCrossCorrSame_NormLevel_ (4)
    - nppiCrossCorrValid_NormLevel_ (4)
  SRC+STEP, SIZE, ptr:src, MISC:i32, SIZE, DST+STEP, ptr:dst:
    - nppiCrossCorrFull_NormLevel_ (20)
    - nppiCrossCorrSame_NormLevel_ (20)
    - nppiCrossCorrValid_NormLevel_ (20)
  SRC+STEP, SIZE, ptr:src, MISC:i32, SIZE, DST+STEP, ptr:dst, ptr:dst:
    - nppiCrossCorrFull_NormLevelAdvanced_ (15)
    - nppiCrossCorrSame_NormLevelAdvanced_ (15)
    - nppiCrossCorrValid_NormLevelAdvanced_ (15)
  SRC+STEP, ptr:dst, MISC:i32, SIZE:
    - nppiAddSquare_ (3)
    - nppiAdd_ (4)
    - nppiAnd_ (12)
    - nppiDiv_ (4)
    - nppiMaxEvery_ (16)
    - nppiMinEvery_ (16)
    - nppiMulScale_ (8)
    - nppiMul_ (4)
    - nppiOr_ (12)
    - nppiSub_ (4)
    - nppiXor_ (12)
  SRC+STEP, ptr:dst, MISC:i32, SIZE, CONST_SCALAR:
    - nppiAdd_ (14)
    - nppiDiv_ (14)
    - nppiMul_ (14)
    - nppiSub_ (15)
  SRC+STEP, ptr:dst, MISC:i32, SIZE, MISC:Npp32f:
    - nppiAddWeighted_ (3)
  SRC+STEP, ptr:dst, MISC:i32, SIZE, MISC:NppRoundMode, CONST_SCALAR:
    - nppiDiv_Round_ (12)
  SRC+STEP, ptr:dst, MISC:i32, ptr:dst, MISC:i32, SIZE:
    - nppiCbYCr (1)
    - nppiYCbCr (8)
    - nppiYCrCb (2)
  SRC+STEP, ptr:src, DST+STEP, SIZE, MISC:Npp32f:
    - nppiCompareEqualEpsC_ (3)
  SRC+STEP, ptr:src, DST+STEP, SIZE, MISC:NppCmpOp:
    - nppiCompareC_ (12)
  SRC+STEP, ptr:src, MISC:i32, DST+STEP, SIZE, RECT:
    - nppiRectStdDev_ (2)
  SRC+STEP, ptr:src, MISC:i32, DST+STEP, SIZE, RECT, CONST_SCALAR:
    - nppiRectStdDev_ (1)
  SRC+STEP, ptr:src, MISC:i32, SIZE, MISC:i32, ptr:dst, ptr:dst:
    - nppiMean_ (4)
    - nppiNorm_Inf_ (4)
    - nppiNorm_L (8)
  SRC+STEP, ptr:src, MISC:i32, SIZE, MISC:i32, ptr:dst, ptr:dst, MISC:ptr, MISC:ptr, ptr:dst:
    - nppiMinMaxIndx_ (4)
  SRC+STEP, ptr:src, MISC:i32, SIZE, MISC:i32, ptr:dst, ptr:dst, ptr:dst:
    - nppiMean_StdDev_ (4)
  SRC+STEP, ptr:src, MISC:i32, SIZE, ptr:dst, ptr:dst:
    - nppiMean_ (4)
    - nppiNorm_Inf_ (4)
    - nppiNorm_L (8)
  SRC+STEP, ptr:src, MISC:i32, SIZE, ptr:dst, ptr:dst, MISC:ptr, MISC:ptr, ptr:dst:
    - nppiMinMaxIndx_ (4)
  SRC+STEP, ptr:src, MISC:i32, SIZE, ptr:dst, ptr:dst, ptr:dst:
    - nppiMean_StdDev_ (4)
  SRC+STEP, ptr:src, MISC:i32, ptr:dst, MISC:i32, SIZE:
    - nppiAddSquare_ (3)
  SRC+STEP, ptr:src, MISC:i32, ptr:dst, MISC:i32, SIZE, MISC:Npp32f:
    - nppiAddWeighted_ (3)
  ptr:dst, MISC:Npp32s, MISC:ptr, MISC:Npp32s, MISC:ptr, MISC:Npp32s, MISC:ptr, MISC:ptr, MISC:ptr, ptr:dst, ptr:dst, ptr:dst, MISC:Npp32u, MISC:Npp32u, MISC:Npp32u, MISC:Npp32u, MISC:ptr, MISC:ptr, SIZE:
    - nppiContoursImageMarchingSquaresInterpolation_ (2)
  ptr:dst, MISC:Npp32s, SIZE, MISC:c_uint, MISC:ptr, ptr:dst, MISC:Npp32s, MISC:ptr, MISC:Npp32s, MISC:ptr, ptr:dst, ptr:dst, ptr:dst, ptr:dst:
    - nppiCompressedMarkerLabelsUFInfo_ (1)
  ptr:dst, MISC:Npp32s, ptr:dst, MISC:Npp32s, MISC:NppiNorm, MISC:NppiWatershedSegmentBoundaryType, SIZE, ptr:dst:
    - nppiSegmentWatershed_ (2)
  ptr:dst, MISC:Npp32u, MISC:Npp32u, MISC:Npp32u, MISC:Npp32u, ptr:dst:
    - nppiCompressedMarkerLabelsUFGetContoursBlockSegmentListSize_C (1)
  ptr:dst, MISC:i32, MISC:Npp32s, MISC:Npp32s:
    - nppiEvenLevelsHost_ (1)
  ptr:dst, MISC:i32, POINT, MISC:Npp16u, MISC:Npp16u, MISC:Npp16u, MISC:Npp16u, MISC:NppiNorm, SIZE, MISC:ptr, ptr:dst:
    - nppiFloodFillGradientBoundary_ (1)
    - nppiFloodFillRangeBoundary_ (1)
  ptr:dst, MISC:i32, POINT, MISC:Npp16u, MISC:Npp16u, MISC:Npp16u, MISC:NppiNorm, SIZE, MISC:ptr, ptr:dst:
    - nppiFloodFillGradient_ (1)
    - nppiFloodFillRange_ (1)
  ptr:dst, MISC:i32, POINT, MISC:Npp16u, MISC:Npp16u, MISC:NppiNorm, SIZE, MISC:ptr, ptr:dst:
    - nppiFloodFillBoundary_ (1)
  ptr:dst, MISC:i32, POINT, MISC:Npp16u, MISC:NppiNorm, SIZE, MISC:ptr, ptr:dst:
    - nppiFloodFill_ (1)
  ptr:dst, MISC:i32, POINT, MISC:Npp32u, MISC:Npp32u, MISC:Npp32u, MISC:Npp32u, MISC:NppiNorm, SIZE, MISC:ptr, ptr:dst:
    - nppiFloodFillGradientBoundary_ (1)
    - nppiFloodFillRangeBoundary_ (1)
  ptr:dst, MISC:i32, POINT, MISC:Npp32u, MISC:Npp32u, MISC:Npp32u, MISC:NppiNorm, SIZE, MISC:ptr, ptr:dst:
    - nppiFloodFillGradient_ (1)
    - nppiFloodFillRange_ (1)
  ptr:dst, MISC:i32, POINT, MISC:Npp32u, MISC:Npp32u, MISC:NppiNorm, SIZE, MISC:ptr, ptr:dst:
    - nppiFloodFillBoundary_ (1)
  ptr:dst, MISC:i32, POINT, MISC:Npp32u, MISC:NppiNorm, SIZE, MISC:ptr, ptr:dst:
    - nppiFloodFill_ (1)
  ptr:dst, MISC:i32, POINT, MISC:Npp8u, MISC:Npp8u, MISC:Npp8u, MISC:Npp8u, MISC:NppiNorm, SIZE, MISC:ptr, ptr:dst:
    - nppiFloodFillGradientBoundary_ (1)
    - nppiFloodFillRangeBoundary_ (1)
  ptr:dst, MISC:i32, POINT, MISC:Npp8u, MISC:Npp8u, MISC:Npp8u, MISC:NppiNorm, SIZE, MISC:ptr, ptr:dst:
    - nppiFloodFillGradient_ (1)
    - nppiFloodFillRange_ (1)
  ptr:dst, MISC:i32, POINT, MISC:Npp8u, MISC:Npp8u, MISC:NppiNorm, SIZE, MISC:ptr, ptr:dst:
    - nppiFloodFillBoundary_ (1)
  ptr:dst, MISC:i32, POINT, MISC:Npp8u, MISC:NppiNorm, SIZE, MISC:ptr, ptr:dst:
    - nppiFloodFill_ (1)
  ptr:dst, MISC:i32, POINT, ptr:dst, ptr:dst, ptr:src, MISC:NppiNorm, SIZE, MISC:ptr, ptr:dst:
    - nppiFloodFillGradient_ (3)
    - nppiFloodFillRange_ (3)
  ptr:dst, MISC:i32, POINT, ptr:dst, ptr:dst, ptr:src, ptr:src, MISC:NppiNorm, SIZE, MISC:ptr, ptr:dst:
    - nppiFloodFillGradientBoundary_ (3)
    - nppiFloodFillRangeBoundary_ (3)
  ptr:dst, MISC:i32, POINT, ptr:src, MISC:NppiNorm, SIZE, MISC:ptr, ptr:dst:
    - nppiFloodFill_ (3)
  ptr:dst, MISC:i32, POINT, ptr:src, ptr:src, MISC:NppiNorm, SIZE, MISC:ptr, ptr:dst:
    - nppiFloodFillBoundary_ (3)
  ptr:dst, MISC:i32, SIZE:
    - nppiAbs_ (8)
    - nppiAlphaPremul_ (2)
    - nppiExp_ (2)
    - nppiGammaFwd_ (2)
    - nppiGammaInv_ (2)
    - nppiLn_ (2)
    - nppiNot_ (4)
    - nppiSqr_ (4)
    - nppiSqrt_ (4)
  ptr:dst, MISC:i32, SIZE, CONST_SCALAR:
    - nppiExp_ (6)
    - nppiLn_ (6)
    - nppiSqr_ (12)
    - nppiSqrt_ (9)
  ptr:dst, MISC:i32, SIZE, MISC:Npp16s:
    - nppiThreshold_GT_ (1)
    - nppiThreshold_LT_ (1)
  ptr:dst, MISC:i32, SIZE, MISC:Npp16s, MISC:Npp16s:
    - nppiThreshold_GTVal_ (1)
    - nppiThreshold_LTVal_ (1)
  ptr:dst, MISC:i32, SIZE, MISC:Npp16s, MISC:Npp16s, MISC:Npp16s, MISC:Npp16s:
    - nppiThreshold_LTValGTVal_ (1)
  ptr:dst, MISC:i32, SIZE, MISC:Npp16s, MISC:Npp16s, MISC:NppCmpOp:
    - nppiThreshold_Val_ (1)
  ptr:dst, MISC:i32, SIZE, MISC:Npp16s, MISC:NppCmpOp:
    - nppiThreshold_ (1)
  ptr:dst, MISC:i32, SIZE, MISC:Npp16u:
    - nppiThreshold_GT_ (1)
    - nppiThreshold_LT_ (1)
  ptr:dst, MISC:i32, SIZE, MISC:Npp16u, MISC:Npp16u:
    - nppiThreshold_GTVal_ (1)
    - nppiThreshold_LTVal_ (1)
  ptr:dst, MISC:i32, SIZE, MISC:Npp16u, MISC:Npp16u, MISC:Npp16u, MISC:Npp16u:
    - nppiThreshold_LTValGTVal_ (1)
  ptr:dst, MISC:i32, SIZE, MISC:Npp16u, MISC:Npp16u, MISC:NppCmpOp:
    - nppiThreshold_Val_ (1)
  ptr:dst, MISC:i32, SIZE, MISC:Npp16u, MISC:NppCmpOp:
    - nppiThreshold_ (1)
  ptr:dst, MISC:i32, SIZE, MISC:Npp32f:
    - nppiThreshold_GT_ (1)
    - nppiThreshold_LT_ (1)
  ptr:dst, MISC:i32, SIZE, MISC:Npp32f, MISC:Npp32f:
    - nppiThreshold_GTVal_ (1)
    - nppiThreshold_LTVal_ (1)
  ptr:dst, MISC:i32, SIZE, MISC:Npp32f, MISC:Npp32f, MISC:Npp32f, MISC:Npp32f:
    - nppiThreshold_LTValGTVal_ (1)
  ptr:dst, MISC:i32, SIZE, MISC:Npp32f, MISC:Npp32f, MISC:NppCmpOp:
    - nppiThreshold_Val_ (1)
  ptr:dst, MISC:i32, SIZE, MISC:Npp32f, MISC:NppCmpOp:
    - nppiThreshold_ (1)
  ptr:dst, MISC:i32, SIZE, MISC:Npp8u:
    - nppiThreshold_GT_ (1)
    - nppiThreshold_LT_ (1)
  ptr:dst, MISC:i32, SIZE, MISC:Npp8u, MISC:Npp8u:
    - nppiThreshold_GTVal_ (1)
    - nppiThreshold_LTVal_ (1)
  ptr:dst, MISC:i32, SIZE, MISC:Npp8u, MISC:Npp8u, MISC:Npp8u, MISC:Npp8u:
    - nppiThreshold_LTValGTVal_ (1)
  ptr:dst, MISC:i32, SIZE, MISC:Npp8u, MISC:Npp8u, MISC:NppCmpOp:
    - nppiThreshold_Val_ (1)
  ptr:dst, MISC:i32, SIZE, MISC:Npp8u, MISC:NppCmpOp:
    - nppiThreshold_ (1)
  ptr:dst, MISC:i32, SIZE, MISC:NppiAxis:
    - nppiMirror_ (20)
  ptr:dst, MISC:i32, SIZE, MISC:i32:
    - nppiSwapChannels_ (10)
  ptr:dst, MISC:i32, SIZE, MISC:i32, MISC:i32, ptr:dst:
    - nppiCompressMarkerLabelsUF_ (1)
  ptr:dst, MISC:i32, SIZE, MISC:ptr:
    - nppiColorTwist (18)
    - nppiColorTwist_ (5)
  ptr:dst, MISC:i32, SIZE, MISC:ptr, MISC:ptr, MISC:i32:
    - nppiLUT_ (12)
    - nppiLUT_Cubic_ (12)
    - nppiLUT_Linear_ (12)
  ptr:dst, MISC:i32, SIZE, MISC:ptr, ptr:src:
    - nppiColorTwist (1)
    - nppiColorTwist_ (1)
  ptr:dst, MISC:i32, SIZE, ptr:dst, MISC:ptr, MISC:i32:
    - nppiLUT_Trilinear_ (1)
  ptr:dst, MISC:i32, SIZE, ptr:src:
    - nppiThreshold_GT_ (8)
    - nppiThreshold_LT_ (8)
  ptr:dst, MISC:i32, SIZE, ptr:src, MISC:NppCmpOp:
    - nppiThreshold_ (8)
  ptr:dst, MISC:i32, SIZE, ptr:src, ptr:src:
    - nppiThreshold_GTVal_ (8)
    - nppiThreshold_LTVal_ (8)
  ptr:dst, MISC:i32, SIZE, ptr:src, ptr:src, MISC:NppCmpOp:
    - nppiThreshold_Val_ (8)
  ptr:dst, MISC:i32, SIZE, ptr:src, ptr:src, MISC:i32:
    - nppiLUT_ (4)
    - nppiLUT_Cubic_ (4)
    - nppiLUT_Linear_ (4)
  ptr:dst, MISC:i32, SIZE, ptr:src, ptr:src, ptr:src, ptr:src:
    - nppiThreshold_LTValGTVal_ (8)
  ptr:src, DST+STEP, SIZE:
    - nppiSet_ (31)
  ptr:src, DST+STEP, SIZE, ptr:src, MISC:i32:
    - nppiSet_ (15)
  ptr:src, MISC:Npp32s, POINT, ptr:dst, MISC:Npp32s, SIZE, MISC:Npp32f, MISC:Npp32f, MISC:Npp32f, MISC:Npp32f, MISC:NppiBorderType, ptr:dst:
    - nppiFilterUnsharpBorder_ (16)
  ptr:src, MISC:Npp32s, SIZE, POINT, ptr:dst, MISC:Npp32s, SIZE, MISC:Npp32f, MISC:i32, ptr:src, MISC:NppiBorderType:
    - nppiFilterGaussPyramidLayerDownBorder_ (6)
    - nppiFilterGaussPyramidLayerUpBorder_ (6)
  ptr:src, MISC:Npp32s, SIZE, POINT, ptr:dst, MISC:Npp32s, SIZE, MISC:Npp32s, MISC:Npp32s, MISC:NppiBorderType:
    - nppiSumWindowColumnBorder_ (9)
    - nppiSumWindowRowBorder_ (9)
  ptr:src, MISC:Npp32s, SIZE, POINT, ptr:dst, MISC:Npp32s, SIZE, MISC:NppiBorderType:
    - nppiDilate (6)
    - nppiErode (6)
    - nppiFilterPrewittHorizBorder_ (12)
    - nppiFilterPrewittVertBorder_ (12)
    - nppiFilterRobertsDownBorder_ (12)
    - nppiFilterRobertsUpBorder_ (12)
    - nppiFilterScharrHorizBorder_ (3)
    - nppiFilterScharrVertBorder_ (3)
    - nppiFilterSharpenBorder_ (16)
    - nppiFilterSobelHorizBorder_ (12)
    - nppiFilterSobelVertBorder_ (12)
  ptr:src, MISC:Npp32s, SIZE, POINT, ptr:dst, MISC:Npp32s, SIZE, MISC:NppiMaskSize, MISC:NppiBorderType:
    - nppiFilterGaussBorder_ (16)
    - nppiFilterHighPassBorder_ (16)
    - nppiFilterLaplaceBorder_ (14)
    - nppiFilterLowPassBorder_ (16)
    - nppiFilterSobelCrossBorder_ (3)
    - nppiFilterSobelHorizBorder_ (2)
    - nppiFilterSobelHorizMaskBorder_ (1)
    - nppiFilterSobelHorizSecondBorder_ (3)
    - nppiFilterSobelVertBorder_ (2)
    - nppiFilterSobelVertMaskBorder_ (1)
    - nppiFilterSobelVertSecondBorder_ (3)
  ptr:src, MISC:Npp32s, SIZE, POINT, ptr:dst, MISC:Npp32s, SIZE, MISC:i32, MISC:i32, MISC:Npp32f, MISC:Npp32f, MISC:NppiBorderType:
    - nppiFilterBilateralGaussBorder_ (6)
  ptr:src, MISC:Npp32s, SIZE, POINT, ptr:dst, MISC:Npp32s, SIZE, MISC:i32, ptr:src, MISC:NppiBorderType:
    - nppiFilterGaussAdvancedBorder_ (16)
  ptr:src, MISC:Npp32s, SIZE, POINT, ptr:dst, MISC:Npp32s, SIZE, SIZE, MISC:Npp32f, MISC:Npp8u, MISC:Npp8u, MISC:NppiBorderType:
    - nppiFilterThresholdAdaptiveBoxBorder_ (1)
  ptr:src, MISC:Npp32s, SIZE, POINT, ptr:dst, MISC:Npp32s, SIZE, SIZE, POINT, MISC:NppiBorderType:
    - nppiFilterBoxBorder_ (16)
    - nppiFilterMaxBorder_ (16)
    - nppiFilterMinBorder_ (16)
  ptr:src, MISC:Npp32s, SIZE, POINT, ptr:dst, MISC:Npp32s, SIZE, SIZE, POINT, MISC:NppiBorderType, ptr:dst:
    - nppiFilterBoxBorderAdvanced_ (13)
  ptr:src, MISC:Npp32s, SIZE, POINT, ptr:dst, MISC:Npp32s, SIZE, SIZE, POINT, ptr:dst, MISC:NppiBorderType:
    - nppiFilterMedianBorder_ (16)
    - nppiFilterWienerBorder_ (12)
  ptr:src, MISC:Npp32s, SIZE, POINT, ptr:dst, MISC:Npp32s, SIZE, ptr:src, MISC:Npp32s, MISC:Npp32s, MISC:Npp32s, MISC:NppiBorderType:
    - nppiFilterColumnBorder_ (12)
    - nppiFilterRowBorder_ (12)
  ptr:src, MISC:Npp32s, SIZE, POINT, ptr:dst, MISC:Npp32s, SIZE, ptr:src, MISC:Npp32s, MISC:Npp32s, MISC:NppiBorderType:
    - nppiFilterColumnBorder_ (4)
    - nppiFilterRowBorder_ (4)
  ptr:src, MISC:Npp32s, SIZE, POINT, ptr:dst, MISC:Npp32s, SIZE, ptr:src, SIZE, POINT, MISC:Npp32s, MISC:NppiBorderType:
    - nppiFilterBorder_ (12)
  ptr:src, MISC:Npp32s, SIZE, POINT, ptr:dst, MISC:Npp32s, SIZE, ptr:src, SIZE, POINT, MISC:NppiBorderType:
    - nppiDilateBorder_ (6)
    - nppiErodeBorder_ (6)
    - nppiFilterBorder_ (5)
    - nppiGrayDilateBorder_ (2)
    - nppiGrayErodeBorder_ (2)
  ptr:src, MISC:Npp32s, ptr:dst, MISC:Npp32s, SIZE:
    - nppiDilate (7)
    - nppiErode (7)
    - nppiFilterPrewittHoriz_ (12)
    - nppiFilterPrewittVert_ (12)
    - nppiFilterRobertsDown_ (12)
    - nppiFilterRobertsUp_ (12)
    - nppiFilterScharrHoriz_ (3)
    - nppiFilterScharrVert_ (3)
    - nppiFilterSharpen_ (16)
    - nppiFilterSobelHoriz_ (12)
    - nppiFilterSobelVert_ (12)
  ptr:src, MISC:Npp32s, ptr:dst, MISC:Npp32s, SIZE, MISC:Npp32s, MISC:Npp32s:
    - nppiSumWindowColumn_ (9)
    - nppiSumWindowRow_ (9)
  ptr:src, MISC:Npp32s, ptr:dst, MISC:Npp32s, SIZE, MISC:NppiMaskSize:
    - nppiFilterGauss_ (16)
    - nppiFilterHighPass_ (16)
    - nppiFilterLaplace_ (14)
    - nppiFilterLowPass_ (16)
    - nppiFilterSobelCross_ (3)
    - nppiFilterSobelHorizMask_ (1)
    - nppiFilterSobelHorizSecond_ (3)
    - nppiFilterSobelHoriz_ (2)
    - nppiFilterSobelVertMask_ (1)
    - nppiFilterSobelVertSecond_ (3)
    - nppiFilterSobelVert_ (2)
  ptr:src, MISC:Npp32s, ptr:dst, MISC:Npp32s, SIZE, MISC:i32, ptr:src:
    - nppiFilterGaussAdvanced_ (16)
  ptr:src, MISC:Npp32s, ptr:dst, MISC:Npp32s, SIZE, SIZE, POINT:
    - nppiFilterBox_ (17)
    - nppiFilterMax_ (16)
    - nppiFilterMin_ (16)
  ptr:src, MISC:Npp32s, ptr:dst, MISC:Npp32s, SIZE, SIZE, POINT, ptr:dst:
    - nppiFilterMedian_ (16)
  ptr:src, MISC:Npp32s, ptr:dst, MISC:Npp32s, SIZE, ptr:src, MISC:Npp32s, MISC:Npp32s:
    - nppiFilterColumn (12)
    - nppiFilterColumn_ (5)
    - nppiFilterRow (12)
    - nppiFilterRow_ (5)
  ptr:src, MISC:Npp32s, ptr:dst, MISC:Npp32s, SIZE, ptr:src, MISC:Npp32s, MISC:Npp32s, MISC:Npp32s:
    - nppiFilterColumn_ (12)
    - nppiFilterRow_ (12)
  ptr:src, MISC:Npp32s, ptr:dst, MISC:Npp32s, SIZE, ptr:src, SIZE, POINT:
    - nppiDilate_ (6)
    - nppiErode_ (6)
    - nppiFilter_ (6)
  ptr:src, MISC:Npp32s, ptr:dst, MISC:Npp32s, SIZE, ptr:src, SIZE, POINT, MISC:Npp32s:
    - nppiFilter_ (12)
  ptr:src, MISC:i32, DST+STEP, SIZE:
    - nppiAlphaPremul_ (2)
  ptr:src, MISC:i32, DST+STEP, SIZE, MISC:Npp16u:
    - nppiAbsDiffC_ (1)
  ptr:src, MISC:i32, DST+STEP, SIZE, MISC:Npp32f:
    - nppiAbsDiffC_ (1)
  ptr:src, MISC:i32, DST+STEP, SIZE, MISC:Npp8u:
    - nppiAbsDiffC_ (1)
  ptr:src, MISC:i32, DST+STEP, SIZE, ptr:dst:
    - nppiAbsDiffDeviceC_ (3)
  ptr:src, MISC:i32, MISC:Npp16s, DST+STEP, SIZE, CONST_SCALAR:
    - nppiAddC_ (1)
    - nppiDivC_ (1)
    - nppiMulC_ (1)
    - nppiSubC_ (1)
  ptr:src, MISC:i32, MISC:Npp16s, ptr:src, MISC:i32, MISC:Npp16s, DST+STEP, SIZE, MISC:NppiAlphaOp:
    - nppiAlphaCompC_ (1)
  ptr:src, MISC:i32, MISC:Npp16u, DST+STEP, SIZE:
    - nppiAlphaPremulC_ (4)
    - nppiAndC_ (1)
    - nppiMulCScale_ (1)
    - nppiOrC_ (1)
    - nppiXorC_ (1)
  ptr:src, MISC:i32, MISC:Npp16u, DST+STEP, SIZE, CONST_SCALAR:
    - nppiAddC_ (1)
    - nppiDivC_ (1)
    - nppiMulC_ (1)
    - nppiSubC_ (1)
  ptr:src, MISC:i32, MISC:Npp16u, ptr:src, MISC:i32, MISC:Npp16u, DST+STEP, SIZE, MISC:NppiAlphaOp:
    - nppiAlphaCompC_ (4)
  ptr:src, MISC:i32, MISC:Npp32f, DST+STEP, SIZE:
    - nppiAddC_ (1)
    - nppiDivC_ (1)
    - nppiMulC_ (1)
    - nppiSubC_ (1)
  ptr:src, MISC:i32, MISC:Npp32f, ptr:src, MISC:i32, MISC:Npp32f, DST+STEP, SIZE, MISC:NppiAlphaOp:
    - nppiAlphaCompC_ (1)
  ptr:src, MISC:i32, MISC:Npp32s, DST+STEP, SIZE:
    - nppiAndC_ (1)
    - nppiOrC_ (1)
    - nppiXorC_ (1)
  ptr:src, MISC:i32, MISC:Npp32s, DST+STEP, SIZE, CONST_SCALAR:
    - nppiAddC_ (1)
    - nppiDivC_ (1)
    - nppiMulC_ (1)
    - nppiSubC_ (1)
  ptr:src, MISC:i32, MISC:Npp32s, ptr:src, MISC:i32, MISC:Npp32s, DST+STEP, SIZE, MISC:NppiAlphaOp:
    - nppiAlphaCompC_ (1)
  ptr:src, MISC:i32, MISC:Npp32u, DST+STEP, SIZE:
    - nppiLShiftC_ (3)
    - nppiRShiftC_ (5)
  ptr:src, MISC:i32, MISC:Npp32u, ptr:src, MISC:i32, MISC:Npp32u, DST+STEP, SIZE, MISC:NppiAlphaOp:
    - nppiAlphaCompC_ (1)
  ptr:src, MISC:i32, MISC:Npp8s, ptr:src, MISC:i32, MISC:Npp8s, DST+STEP, SIZE, MISC:NppiAlphaOp:
    - nppiAlphaCompC_ (1)
  ptr:src, MISC:i32, MISC:Npp8u, DST+STEP, SIZE:
    - nppiAlphaPremulC_ (4)
    - nppiAndC_ (1)
    - nppiMulCScale_ (1)
    - nppiOrC_ (1)
    - nppiXorC_ (1)
  ptr:src, MISC:i32, MISC:Npp8u, DST+STEP, SIZE, CONST_SCALAR:
    - nppiAddC_ (1)
    - nppiDivC_ (1)
    - nppiMulC_ (1)
    - nppiSubC_ (1)
  ptr:src, MISC:i32, MISC:Npp8u, ptr:src, MISC:i32, MISC:Npp8u, DST+STEP, SIZE, MISC:NppiAlphaOp:
    - nppiAlphaCompC_ (4)
  ptr:src, MISC:i32, MISC:Npp8u, ptr:src, MISC:i32, MISC:Npp8u, DST+STEP, SIZE, ptr:dst, MISC:NppiAlphaOp:
    - nppiAlphaCompColorKey_ (1)
  ptr:src, MISC:i32, ptr:src, DST+STEP, SIZE:
    - nppiAddC_ (3)
    - nppiAddDeviceC_ (4)
    - nppiAndC_ (9)
    - nppiDivC_ (3)
    - nppiDivDeviceC_ (4)
    - nppiLShiftC_ (9)
    - nppiMulCScale_ (6)
    - nppiMulC_ (3)
    - nppiMulDeviceCScale_ (8)
    - nppiMulDeviceC_ (4)
    - nppiOrC_ (9)
    - nppiRShiftC_ (15)
    - nppiSubC_ (3)
    - nppiSubDeviceC_ (4)
    - nppiXorC_ (9)
  ptr:src, MISC:i32, ptr:src, DST+STEP, SIZE, CONST_SCALAR:
    - nppiAddC_ (10)
    - nppiAddDeviceC_ (14)
    - nppiDivC_ (10)
    - nppiDivDeviceC_ (14)
    - nppiMulC_ (10)
    - nppiMulDeviceC_ (14)
    - nppiSubC_ (10)
    - nppiSubDeviceC_ (14)
  ptr:src, MISC:i32, ptr:src, MISC:i32, DST+STEP, SIZE:
    - nppiAbsDiff_ (5)
    - nppiAdd_ (5)
    - nppiAnd_ (12)
    - nppiDiv_ (5)
    - nppiMulScale_ (8)
    - nppiMul_ (5)
    - nppiOr_ (12)
    - nppiSub_ (5)
    - nppiXor_ (12)
    - nppiYCbCr (11)
  ptr:src, MISC:i32, ptr:src, MISC:i32, DST+STEP, SIZE, CONST_SCALAR:
    - nppiAdd_ (14)
    - nppiDiv_ (14)
    - nppiMul_ (14)
    - nppiSub_ (15)
  ptr:src, MISC:i32, ptr:src, MISC:i32, DST+STEP, SIZE, MISC:Npp32f:
    - nppiCompareEqualEps_ (4)
  ptr:src, MISC:i32, ptr:src, MISC:i32, DST+STEP, SIZE, MISC:Npp8u:
    - nppiCompColorKey_ (1)
  ptr:src, MISC:i32, ptr:src, MISC:i32, DST+STEP, SIZE, MISC:NppCmpOp:
    - nppiCompare_ (16)
  ptr:src, MISC:i32, ptr:src, MISC:i32, DST+STEP, SIZE, MISC:NppRoundMode, CONST_SCALAR:
    - nppiDiv_Round_ (12)
  ptr:src, MISC:i32, ptr:src, MISC:i32, DST+STEP, SIZE, MISC:NppiAlphaOp:
    - nppiAlphaComp_ (12)
  ptr:src, MISC:i32, ptr:src, MISC:i32, DST+STEP, SIZE, ptr:dst:
    - nppiCompColorKey_ (2)
  ptr:src, MISC:i32, ptr:src, MISC:i32, SIZE, ptr:dst, ptr:dst:
    - nppiAverageError_ (32)
    - nppiAverageRelativeError_ (32)
    - nppiDotProd_ (28)
    - nppiMSE_ (2)
    - nppiMSSSIM_ (1)
    - nppiMaximumError_ (32)
    - nppiMaximumRelativeError_ (32)
    - nppiNormDiff_Inf_ (16)
    - nppiNormDiff_L (32)
    - nppiNormRel_Inf_ (16)
    - nppiNormRel_L (32)
    - nppiPSNR_ (2)
    - nppiQualityIndex_ (9)
    - nppiSSIM_ (2)
    - nppiWMSSSIM_ (2)
  ptr:src, MISC:i32, ptr:src, MISC:i32, ptr:dst, MISC:i32, SIZE:
    - nppiAddProduct_ (3)
  ptr:src, MISC:i32, ptr:src, MISC:i32, ptr:src, MISC:i32, SIZE, MISC:i32, ptr:dst, ptr:dst:
    - nppiNormDiff_Inf_ (4)
    - nppiNormDiff_L (8)
    - nppiNormRel_Inf_ (4)
    - nppiNormRel_L (8)
  ptr:src, MISC:i32, ptr:src, MISC:i32, ptr:src, MISC:i32, SIZE, ptr:dst, ptr:dst:
    - nppiNormDiff_Inf_ (4)
    - nppiNormDiff_L (8)
    - nppiNormRel_Inf_ (4)
    - nppiNormRel_L (8)
  ptr:src, MISC:i32, ptr:src, MISC:i32, ptr:src, MISC:i32, ptr:dst, MISC:i32, SIZE:
    - nppiAddProduct_ (3)
  ptr:src, MISC:ptr, MISC:i32, SIZE:
    - nppiAddC_ (2)
    - nppiAddDeviceC_ (3)
    - nppiDivC_ (2)
    - nppiDivDeviceC_ (3)
    - nppiMulC_ (2)
    - nppiMulDeviceC_ (3)
    - nppiSubC_ (2)
    - nppiSubDeviceC_ (3)
  ptr:src, SIZE, MISC:i32, RECT, DST+STEP, RECT, MISC:f64, MISC:f64, MISC:f64, INTERP:
    - nppiRotate_ (12)
  ptr:src, SIZE, MISC:i32, RECT, DST+STEP, RECT, MISC:f64, MISC:f64, MISC:f64, MISC:f64, INTERP:
    - nppiResizeSqrPixel_ (20)
  ptr:src, SIZE, MISC:i32, RECT, DST+STEP, RECT, MISC:f64, MISC:f64, ptr:dst, INTERP:
    - nppiResizeSqrPixel_ (1)
  ptr:src, SIZE, MISC:i32, RECT, DST+STEP, RECT, MISC:ptr, INTERP:
    - nppiWarpAffineBack_ (16)
    - nppiWarpAffine_ (20)
    - nppiWarpPerspectiveBack_ (16)
    - nppiWarpPerspective_ (16)
  ptr:src, SIZE, MISC:i32, RECT, MISC:ptr, DST+STEP, RECT, MISC:ptr, INTERP:
    - nppiWarpAffineQuad_ (16)
    - nppiWarpPerspectiveQuad_ (16)
  ptr:src, SIZE, MISC:i32, RECT, ptr:src, MISC:i32, ptr:src, MISC:i32, DST+STEP, SIZE, INTERP:
    - nppiRemap_ (20)
  ptr:src, ptr:dst, MISC:i32, SIZE:
    - nppiAddC_ (3)
    - nppiAddDeviceC_ (4)
    - nppiAndC_ (9)
    - nppiDivC_ (3)
    - nppiDivDeviceC_ (4)
    - nppiLShiftC_ (9)
    - nppiMulCScale_ (6)
    - nppiMulC_ (3)
    - nppiMulDeviceCScale_ (8)
    - nppiMulDeviceC_ (4)
    - nppiOrC_ (9)
    - nppiRShiftC_ (15)
    - nppiSubC_ (3)
    - nppiSubDeviceC_ (4)
    - nppiXorC_ (9)
  ptr:src, ptr:dst, MISC:i32, SIZE, CONST_SCALAR:
    - nppiAddC_ (10)
    - nppiAddDeviceC_ (14)
    - nppiDivC_ (10)
    - nppiDivDeviceC_ (14)
    - nppiMulC_ (10)
    - nppiMulDeviceC_ (14)
    - nppiSubC_ (10)
    - nppiSubDeviceC_ (14)

== RESIZE SANITY CHECK ==
  nppiResize_8u_C1R -> SRC+STEP, SIZE, RECT, DST+STEP, SIZE, RECT, INTERP
  OK[!]
