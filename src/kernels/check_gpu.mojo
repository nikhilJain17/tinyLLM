from std.gpu.host import DeviceContext
from layout.tile_layout import row_major
from layout import TileTensor
from matmul import naive_matmul, single_block_shm_matmul, tiled_matmul, dtype, SIZE, TILE, layout, LayoutType


comptime kernel = tiled_matmul[SIZE]
comptime grid_dim = (SIZE // TILE, SIZE // TILE)   # one block per output tile
comptime block_dim = (TILE, TILE)

# Runner for kernels until we stand up C++ <> Mojo interop.
def main():
    try:
        var ctx = DeviceContext()
        print("GPU:", ctx.name())
        run_matmul(ctx)
    except e:
        print("error:", e)

def run_matmul(ctx: DeviceContext) raises:
    var out = ctx.enqueue_create_buffer[dtype](SIZE * SIZE)
    out.enqueue_fill(0)
    var a_buf = ctx.enqueue_create_buffer[dtype](SIZE * SIZE)
    var b_buf = ctx.enqueue_create_buffer[dtype](SIZE * SIZE)

    # Oracle: A = distinct values (0,1,2,...), B = identity.  C = A·I must equal A.
    with a_buf.map_to_host() as a_host, b_buf.map_to_host() as b_host:
        for r in range(SIZE):
            for c in range(SIZE):
                a_host[r * SIZE + c] = Scalar[dtype](r * SIZE + c)
                b_host[r * SIZE + c] = Scalar[dtype](1) if r == c else Scalar[dtype](0)

    var out_tensor = TileTensor(out, layout)
    var a_tensor = TileTensor[mut=False, dtype, LayoutType](a_buf, layout)
    var b_tensor = TileTensor[mut=False, dtype, LayoutType](b_buf, layout)

    ctx.enqueue_function[kernel](
        out_tensor,
        a_tensor,
        b_tensor,
        grid_dim=grid_dim,
        block_dim=block_dim,
    )
    ctx.synchronize()

    print("=== OUT (should equal A: 0,1,2,...) ===")
    var ok = True
    with out.map_to_host() as out_host:
        for r in range(SIZE):
            for c in range(SIZE):
                var got = out_host[r * SIZE + c]
                var want = Scalar[dtype](r * SIZE + c)
                print(got, end=" ")
                if got != want:
                    ok = False
            print()

    if ok:
        print("PASS  C == A  — tiling is correct")
    else:
        print("FAIL  output != A  — index/tiling bug")
