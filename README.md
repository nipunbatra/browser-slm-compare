# Browser model comparison

A static GitHub Pages demo that sends the same prompt to Gemma 3 270M, SmolLM2 360M Instruct, and Liquid LFM2.5 350M. The browser downloads public model files on demand and runs inference locally. There is no inference API, account, or prompt telemetry.

## Run locally

Start the local server from the repository root.

```sh
python3 -m http.server 8765 --bind 127.0.0.1 --directory docs
```

Open `http://127.0.0.1:8765/` in a current browser with WebGPU support. The first run for all three models reads about 1.03 GB of model and tokenizer files, plus the runtime. Select a single model to reduce downloads and memory pressure. Models and backends run in separate, sequential workers. The site may reuse downloads via the browser cache.

**Run prompt** sends one shared prompt to the selected models. **Run 5 checks** runs the two India phrasings, France, arithmetic, and one-word instruction prompts. All generations start from a single user message and the tokenizer’s own chat template. Greedy decoding uses one beam and 64 maximum new tokens, with end tokens inherited from each checkpoint.

**WebGPU + WASM** runs both providers where supported, using the same weights, tokenizer, and generation settings. The WebGPU label reports whether the worker observed compute dispatches during generation. ONNX Runtime may execute individual operators on the CPU even with WebGPU enabled. WASM runs SmolLM2 on the CPU in a single thread and requires selecting a WASM option. **Export JSON** saves each run locally, including the source revision, weight filenames and upstream SHA-256 hashes, requested and observed backend, formatted input, output token IDs, decoded text with and without special tokens, and timing.

## Pinned artifacts

Both providers use the same `q4` ONNX graph for a given model. This is a 4-bit weight quantization, with unquantized tensors where applicable. Full file sizes and hashes are in [the model manifest](docs/models.js).

| Model | Hugging Face revision | Weights | Terms |
| --- | --- | --- | --- |
| Gemma 3 270M Instruct | [onnx-community/gemma-3-270m-it-ONNX @ 2dbbfdb1b59bd034eb959428c6a7da9dd7ea27f0](https://huggingface.co/onnx-community/gemma-3-270m-it-ONNX/tree/2dbbfdb1b59bd034eb959428c6a7da9dd7ea27f0) | `model_q4.onnx` + `model_q4.onnx_data` | [Gemma Terms](https://gemma.ai/resources/terms/) |
| SmolLM2 360M Instruct | [HuggingFaceTB/SmolLM2-360M-Instruct @ a10cc1512eabd3dde888204e902eca88bddb4951](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct/tree/a10cc1512eabd3dde888204e902eca88bddb4951) | `model_q4.onnx` | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| Liquid LFM2.5 350M | [LiquidAI/LFM2.5-350M-ONNX @ d11593fd9eb408e322667926656598896c2d5ff9](https://huggingface.co/LiquidAI/LFM2.5-350M-ONNX/tree/d11593fd9eb408e322667926656598896c2d5ff9) | `model_q4.onnx` + `model_q4.onnx_data` | [LFM 1.0](https://huggingface.co/LiquidAI/LFM2.5-350M-ONNX/blob/d11593fd9eb408e322667926656598896c2d5ff9/LICENSE) |

[Transformers.js 4.3.2](https://github.com/huggingface/transformers.js/releases/tag/4.3.2) uses ONNX Runtime Web `1.31.0-dev.20260914-8d85527a0`. The runtime is fetched from jsDelivr. This repository publishes the application and synthetic prompts and links to the upstream terms for each model download.

## Desktop smoke test

Chrome 157 on a Mac Studio generated every response below. GPU compute dispatches were observed for all 15 WebGPU completions. The table abbreviates sentence-length replies. The app displays the full text and exports the token IDs.

| Prompt | Gemma 3 270M | SmolLM2 360M | Liquid LFM2.5 350M |
| --- | --- | --- | --- |
| Who is the capital of India? | New Delhi district | New Delhi | New Delhi |
| What is the capital of India? | Yellowstone region | New Delhi | New Delhi |
| What is the capital of France? | Paris | Paris | Paris |
| What is 2+2? | 2 | 4 in a reply that reached the token limit | 2 + 2 = 4 |
| Reply with exactly one word: mango | `Mango.\n` | `Mango.` | `mango` |

SmolLM2 produced identical WebGPU and WASM token IDs for all five prompts. Gemma and Liquid require `GatherBlockQuantized`, which the pinned WASM runtime lacks.

A separate Mac Studio test used Transformers.js 4.3.2 with native ONNX Runtime Node 1.30.0 on the CPU. It used the same pinned ONNX files, rendered prompts, and greedy decoding settings. The generated token IDs matched Chrome WebGPU for all 15 model and prompt combinations, including both Gemma errors in the table.

Physical iPhone Safari and native desktop Safari are unverified.

## Deploy

Publish the `main` branch’s `/docs` folder using [GitHub Pages](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site). No build step is required.
