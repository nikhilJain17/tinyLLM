from std.gpu.host import DeviceContext
from layout.tile_layout import row_major
from layout import TileTensor
from matmul import matmul, dtype, SIZE, layout, LayoutType


comptime kernel = matmul[SIZE]
comptime BLOCKS_PER_GRID = (1, 1)
comptime THREADS_PER_BLOCK = (SIZE, SIZE)
# Runner for kernels until we stand up C++ <> Mojo interop.
def main():
    try:
        var ctx = DeviceContext()
        print("GPU:", ctx.name())
        run_matmul(ctx)
    except e:
        pass

def run_matmul(ctx: DeviceContext) raises:
    var out = ctx.enqueue_create_buffer[dtype](SIZE * SIZE)
    out.enqueue_fill(0)
    var inp1 = ctx.enqueue_create_buffer[dtype](SIZE * SIZE)
    inp1.enqueue_fill(5)
    var inp2 = ctx.enqueue_create_buffer[dtype](SIZE * SIZE)
    inp2.enqueue_fill(2)
    
    var out_tensor = TileTensor(out, layout)
    var a_tensor = TileTensor[mut=False, dtype, LayoutType](inp1, layout)
    var b_tensor = TileTensor[mut=False, dtype, LayoutType](inp2, layout)    


    ctx.enqueue_function[kernel](
        out_tensor,
        a_tensor,
        b_tensor,
        grid_dim=BLOCKS_PER_GRID,
        block_dim=THREADS_PER_BLOCK,
    )

    print("==== A ====")
    with inp1.map_to_host() as a_host:
        for r in range(SIZE):
            for c in range(SIZE):
                print(a_host[r * SIZE + c], end=" ")
            print()

    print("==== B ====")
    with inp2.map_to_host() as b_host:
        for r in range(SIZE):
            for c in range(SIZE):
                print(b_host[r * SIZE + c], end=" ")
            print()

    print("=== OUT ===")
    with out.map_to_host() as out_host:
        for r in range(SIZE):
            for c in range(SIZE):
                print(out_host[r * SIZE + c], end=" ")
            print()
