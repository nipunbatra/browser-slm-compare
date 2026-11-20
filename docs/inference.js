import { MAX_NEW_TOKENS, MODELS, ORT_VERSION, TRANSFORMERS_VERSION } from './models.js';

const runtimeUrl = `https://cdn.jsdelivr.net/npm/@huggingface/transformers@${TRANSFORMERS_VERSION}/dist/transformers.min.js`;

function report(type, detail = {}) {
  self.postMessage({ type, ...detail });
}

function observeGpuDispatches() {
  let dispatches = 0;
  const computePass = self.GPUComputePassEncoder?.prototype;
  const methods = ['dispatchWorkgroups', 'dispatchWorkgroupsIndirect'];
  if (!computePass || methods.some((method) => typeof computePass[method] !== 'function')) {
    return null;
  }
  for (const method of methods) {
    const original = computePass[method];
    computePass[method] = function (...args) {
      const result = original.apply(this, args);
      dispatches += 1;
      return result;
    };
  }
  // The app terminates this dedicated worker before starting the next session.
  return () => dispatches;
}

async function configureBackend(env, backend) {
  const ort = env.backends.onnx;
  if (ort.versions.web !== ORT_VERSION) {
    throw new Error(`Expected ONNX Runtime ${ORT_VERSION}, received ${ort.versions.web}.`);
  }
  ort.wasm.numThreads = 1;
  ort.wasm.proxy = false;

  if (backend === 'wasm') return { adapter: null, countDispatches: null };

  if (!self.navigator.gpu) {
    throw new Error('This browser does not expose WebGPU to the inference worker. Choose WASM (CPU) to try the same files.');
  }
  const adapter = await self.navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter || adapter.info?.isFallbackAdapter || adapter.isFallbackAdapter) {
    throw new Error('No hardware WebGPU adapter was available. Choose WASM (CPU) to try the same files.');
  }
  ort.webgpu.adapter = adapter;
  return { adapter, countDispatches: observeGpuDispatches() };
}

async function generate(model, tokenizer, prompt, backend, gpu) {
  const formattedPrompt = tokenizer.apply_chat_template([{ role: 'user', content: prompt }], {
    add_generation_prompt: true,
    tokenize: false,
  });
  const inputs = tokenizer(formattedPrompt, { add_special_tokens: false });
  const inputTokens = inputs.input_ids.dims[1];
  const before = gpu.countDispatches?.();
  const start = performance.now();
  let sequences;
  try {
    sequences = await model.generate({
      ...inputs,
      do_sample: false,
      num_beams: 1,
      max_new_tokens: MAX_NEW_TOKENS,
      return_dict_in_generate: false,
    });
    const elapsedMs = Math.round(performance.now() - start);
    const gpuDispatches = before === undefined ? null : gpu.countDispatches() - before;
    const tokenIds = Array.from(sequences.data.slice(inputTokens), Number);
    const eos = model.generation_config?.eos_token_id ?? model.config.eos_token_id;
    const endTokens = [eos].flat();
    return {
      backendUsed: backend === 'wasm' ? 'WASM (CPU)' :
        gpuDispatches > 0 ? 'WebGPU (GPU compute observed)' : 'WebGPU requested · GPU compute unverified',
      adapter: gpu.adapter ? {
        vendor: gpu.adapter.info?.vendor || null,
        architecture: gpu.adapter.info?.architecture || null,
      } : null,
      gpuDispatches,
      text: tokenizer.decode(tokenIds, { skip_special_tokens: true, clean_up_tokenization_spaces: false }),
      raw: tokenizer.decode(tokenIds, { skip_special_tokens: false, clean_up_tokenization_spaces: false }),
      formattedPrompt,
      tokenIds,
      inputTokens,
      elapsedMs,
      stop: endTokens.includes(tokenIds.at(-1)) ? 'End token' :
        tokenIds.length === MAX_NEW_TOKENS ? 'Token limit' : 'Runtime stop',
    };
  } finally {
    sequences?.dispose();
    for (const tensor of Object.values(inputs)) tensor.dispose();
  }
}

function showProgress() {
  const files = new Map();
  let lastUpdate = 0;
  return (event) => {
    if (event.status !== 'progress' || !Number.isFinite(event.loaded)) return;
    files.set(event.file, event.loaded);
    if (performance.now() - lastUpdate < 180) return;
    lastUpdate = performance.now();
    report('download', { bytes: [...files.values()].reduce((total, bytes) => total + bytes, 0) });
  };
}

self.addEventListener('message', async ({ data }) => {
  const entry = MODELS.find((item) => item.key === data.key);
  const { backend, prompts } = data;
  let model;
  try {
    const { AutoModelForCausalLM, AutoTokenizer, env } = await import(runtimeUrl);
    env.allowLocalModels = false;
    env.useBrowserCache = true;
    const gpu = await configureBackend(env, backend);
    const options = { revision: entry.revision, progress_callback: showProgress() };
    const tokenizer = await AutoTokenizer.from_pretrained(entry.checkpoint, options);
    model = await AutoModelForCausalLM.from_pretrained(entry.checkpoint, {
      ...options,
      dtype: entry.quantization,
      device: backend,
    });
    for (const [index, prompt] of prompts.entries()) {
      report('generating', { index });
      const result = await generate(model, tokenizer, prompt, backend, gpu);
      report('result', { index, result });
    }
  } catch (error) {
    report('failed', { message: error?.message || String(error) });
  } finally {
    try {
      await model?.dispose();
    } finally {
      report('done');
    }
  }
}, { once: true });
