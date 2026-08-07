from std.gpu import thread_idx, block_idx, block_dim, barrier
from layout import TileTensor
from layout.tile_layout import row_major
from layout.tile_tensor import stack_allocation

comptime SIZE = 9 
comptime layout = row_major[SIZE, SIZE]()
comptime dtype = DType.float32
comptime LayoutType = type_of(layout)
comptime TILE = 3
comptime tile_layout = row_major[TILE, TILE]()

def naive_matmul[
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

def single_block_shm_matmul[
    size: Int
](
    output: TileTensor[mut=True, dtype, LayoutType, MutAnyOrigin],
    a: TileTensor[mut=False, dtype, LayoutType, ImmutAnyOrigin],
    b: TileTensor[mut=False, dtype, LayoutType, ImmutAnyOrigin],
):
    M = a.static_shape[0]
    N = b.static_shape[1]
    assert a.static_shape[1] == a.static_shape[0]
    K = a.static_shape[1]
    print("m, n, k", M, N, K)
    var row = block_dim.y * block_idx.y + thread_idx.y
    var col = block_dim.x * block_idx.x + thread_idx.x
    var local_row = thread_idx.y
    var local_col = thread_idx.x
    # 1. load a, b into shared memory
    comptime shared_layout = row_major[SIZE, SIZE]()
    a_shm = stack_allocation[dtype=dtype, address_space=AddressSpace.SHARED](shared_layout)
    b_shm = stack_allocation[dtype=dtype, address_space=AddressSpace.SHARED](shared_layout)
    a_shm[local_row, local_col] = a[row, col]
    b_shm[local_row, local_col] = b[row, col]
    # 2. barrier
    barrier()
    # 3. read from shared memory
    var accum = Scalar[dtype](0)
    for k in range(K):
        accum += a[row, k] * b[k, col]
    output[row, col] = accum

def tiled_matmul[size: Int](
    output: TileTensor[mut=True, dtype, LayoutType, MutAnyOrigin],
    a: TileTensor[mut=False, dtype, LayoutType, ImmutAnyOrigin],
    b: TileTensor[mut=False, dtype, LayoutType, ImmutAnyOrigin],
):
    var local_row = thread_idx.y
    var local_col = thread_idx.x
    var global_row = block_idx.y * TILE + local_row
    var global_col = block_idx.x * TILE + local_col

    var a_shm = stack_allocation[dtype=dtype, address_space=AddressSpace.SHARED](tile_layout)
    var b_shm = stack_allocation[dtype=dtype, address_space=AddressSpace.SHARED](tile_layout)

    var accum = Scalar[dtype](0)

    # walk over K dimension in TILE-sized chunks
    comptime for s in range(SIZE // TILE):
        a_col = UInt(s * TILE) + local_col
        b_row = UInt(s * TILE) + local_row
        a_shm[local_row, local_col] = a[global_row, a_col]
        b_shm[local_row, local_col] = b[b_row, global_col]
    
        barrier()
    
        for t in range(TILE): 
            accum += a_shm[local_row,t] * b_shm[t,local_col]
    
        barrier()
    
    output[global_row, global_col] = accum
