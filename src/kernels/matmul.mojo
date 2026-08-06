from std.gpu import thread_idx, block_idx, block_dim
from layout import TileTensor
from layout.tile_layout import row_major

comptime SIZE = 2
comptime layout = row_major[SIZE, SIZE]()
comptime dtype = DType.float32
comptime LayoutType = type_of(layout)

def matmul[
    size: Int
](
    output: TileTensor[mut=True, dtype, LayoutType, MutAnyOrigin],
    a: TileTensor[mut=False, dtype, LayoutType, MutAnyOrigin],
    b: TileTensor[mut=False, dtype, LayoutType, MutAnyOrigin],
):
    M = a.static_shape[0]
    N = b.static_shape[1]
    assert a.static_shape[1] == a.static_shape[0]
    K = a.static_shape[1]
    print("m, n, k", M, N, K)
    var row = block_dim.y * block_idx.y + thread_idx.y
    var col = block_dim.x * block_idx.x + thread_idx.x
    if row <= M and col <= N:
        var accum = Scalar[dtype](0)
        for k in range(K):
            accum += a[row, k] * b[k, col]
        output[row, col] = accum