[8-7-2026]
(coding)
- shared memory version of naive matmul 
(learnings)
- tiling -> stream weights from global memory to shared memory
    -> if entire matrix is too big to fit in shared memory at once
- manual indexing is not common in kernels; instead we can use higher level apis
(todo)
- "productionize" kernel by
    1. making it work for non square matrices
    2. parameterize on size, add guards around indices
    3. launch with proper weights and blocks and threads per block etc.
- *replace with real tiling etc apis*

[8-6-2026]
(planning)
- starting to implement kernels in mojo
- f32 kernels written in mojo
- target llama-3.2 1b ops first
- do mojo <> c++ interop later
- measure accuracy against pytorch impls
(coding)
- very naive matmul in mojo, getting familiar with syntax etc
(learnings)
- kernels must be parameterized by dtype and shape
    - see reeselevine's pre-wgsl for a good example of dsl automating this
- gpu memory hierarchy: 
    global memory (vram) 
    --> shared memory (shared across threads within streaming multiprocessor)
        --> local memory (thread-specific)
- spawn one thread per _output_, and the kernel launcher is the one that spawns
the number of threads based on tensor shapes 

(todo)
- f16 --> f32 dequant
- embedding table
- c++ <> mojo interop

[7-19-2026]
- bpe vs merge bpe
- weights in gguf come shipped with token vocabulary
- we still need a pre-tokenization regex pass to slice input text into chunks
- spec in on llama vocab and f16

[prev]
- TODO write about weight loader stuff -- chunked vs one shot, fully resident vs mmapp