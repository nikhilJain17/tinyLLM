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