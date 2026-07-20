## Tokenizer
Specifically target gpt2-style BPE vocabularies.
No `tokenizer.ggml.scores`, instead use the provided merge map in `tokenizer.ggml.merges`.
If no merge map, then fall back to normal tiktoken merge order where the vocab order is the merge order.

## Weight Loader
Only target GGUF files.

## Example models
https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF/blob/main/Llama-3.2-1B-Instruct-F16.gguf