export const TRANSFORMERS_VERSION = '4.3.2';
export const ORT_VERSION = '1.31.0-dev.20260914-8d85527a0';
export const MAX_NEW_TOKENS = 64;
export const MAX_PROMPT_LENGTH = 2_000;

const WASM_QUANTIZED_EMBEDDING = 'Unavailable with these files. The pinned WASM runtime lacks the GatherBlockQuantized operator.';

export const CHECKS = [
  { label: 'India · who', prompt: 'Who is the capital of India?' },
  { label: 'India · what', prompt: 'What is the capital of India?' },
  { label: 'France', prompt: 'What is the capital of France?' },
  { label: '2 + 2', prompt: 'What is 2+2?' },
  { label: 'Mango', prompt: 'Reply with exactly one word: mango' },
];

export const MODELS = [
  {
    key: 'gemma',
    name: 'Gemma 3 270M',
    checkpoint: 'onnx-community/gemma-3-270m-it-ONNX',
    revision: '2dbbfdb1b59bd034eb959428c6a7da9dd7ea27f0',
    license: 'Gemma Terms',
    licenseUrl: 'https://gemma.ai/resources/terms/',
    weights: [
      { file: 'onnx/model_q4.onnx', bytes: 242522, sha256: 'cb3615746a988839308507933d4e0501ff695e747a0a0ce15c0ef53083f8ec54' },
      { file: 'onnx/model_q4.onnx_data', bytes: 322933248, sha256: '25c4a14af32fc1011a90504076024b0e1f65f915e9690b44a5484200d2e20851' },
    ],
    downloadMB: 344,
    wasmUnavailable: WASM_QUANTIZED_EMBEDDING,
  },
  {
    key: 'smol',
    name: 'SmolLM2 360M',
    checkpoint: 'HuggingFaceTB/SmolLM2-360M-Instruct',
    revision: 'a10cc1512eabd3dde888204e902eca88bddb4951',
    license: 'Apache 2.0',
    licenseUrl: 'https://www.apache.org/licenses/LICENSE-2.0',
    weights: [
      { file: 'onnx/model_q4.onnx', bytes: 387943246, sha256: 'a4ffda96e65beafc6f6cef0cbcf9fdc1cbdd79c230906bf3897d190547c7a596' },
    ],
    downloadMB: 390,
  },
  {
    key: 'liquid',
    name: 'Liquid LFM2.5 350M',
    checkpoint: 'LiquidAI/LFM2.5-350M-ONNX',
    revision: 'd11593fd9eb408e322667926656598896c2d5ff9',
    license: 'LFM 1.0',
    licenseUrl: 'https://huggingface.co/LiquidAI/LFM2.5-350M-ONNX/blob/d11593fd9eb408e322667926656598896c2d5ff9/LICENSE',
    weights: [
      { file: 'onnx/model_q4.onnx', bytes: 183442, sha256: 'd1a705712e93aafaba1346b32245fa59e7857a46e2272003c0e8c524977e0de8' },
      { file: 'onnx/model_q4.onnx_data', bytes: 293629952, sha256: '71ec6ad38a4c463dcb3dba671d06a1d9861be3a23e51290d818b95c0b7d2a5db' },
    ],
    downloadMB: 297,
    wasmUnavailable: WASM_QUANTIZED_EMBEDDING,
  },
].map((model) => ({ ...model, quantization: 'q4' }));
