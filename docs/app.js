import {
  CHECKS, MAX_NEW_TOKENS, MAX_PROMPT_LENGTH, MODELS, ORT_VERSION, TRANSFORMERS_VERSION,
} from './models.js';

const form = document.querySelector('#run-form');
const controls = document.querySelector('#run-controls');
const promptInput = document.querySelector('#prompt');
const backendInput = document.querySelector('#backend');
const modelOptions = document.querySelector('#model-options');
const checksButton = document.querySelector('#run-checks');
const downloads = document.querySelector('#downloads');
const status = document.querySelector('#run-status');
const stopButton = document.querySelector('#stop');
const exportButton = document.querySelector('#export');
const promptSelect = document.querySelector('#view-prompt');
const shownPrompt = document.querySelector('#shown-prompt');
const modelBodies = new Map();
const backendNames = { webgpu: 'WebGPU', wasm: 'WASM (CPU)' };
let run = null;
let cancelSession = null;

function node(tag, content, className) {
  const element = document.createElement(tag);
  if (content !== undefined) element.textContent = content;
  if (className) element.className = className;
  return element;
}

function link(label, href, className) {
  const anchor = node('a', label, className);
  anchor.href = href;
  anchor.target = '_blank';
  anchor.rel = 'noreferrer';
  return anchor;
}

function selectedModels() {
  return MODELS.filter((model) => modelOptions.querySelector('#select-' + model.key).checked);
}

function updateDownloads() {
  const selected = selectedModels();
  const runnable = backendInput.value === 'wasm' ? selected.filter((model) => !model.wasmUnavailable) : selected;
  const total = runnable.reduce((sum, model) => sum + model.downloadMB, 0);
  downloads.textContent = total
    ? `First run downloads about ${total.toLocaleString()} MB of model files for this selection, plus the runtime. Cached files may be reused.`
    : backendInput.value === 'wasm' ? 'Select SmolLM2 to run WASM, or choose WebGPU.' : 'Select at least one model to run.';
  form.querySelector('button[type="submit"]').disabled = !total;
  checksButton.disabled = !total;
}

function showCapability() {
  const hardware = document.querySelector('#hardware');
  hardware.textContent = navigator.gpu
    ? 'WebGPU is available to this page. The inference worker checks for a hardware adapter.'
    : 'WebGPU is unavailable to this page. You can choose WASM (CPU).';
}

for (const example of CHECKS) {
  const button = node('button', example.label);
  button.type = 'button';
  button.title = example.prompt;
  button.addEventListener('click', () => {
    promptInput.value = example.prompt;
    promptInput.setCustomValidity('');
    promptInput.focus();
  });
  document.querySelector('#examples').append(button);
}

for (const model of MODELS) {
  const label = node('label');
  const checkbox = node('input');
  checkbox.type = 'checkbox';
  checkbox.id = 'select-' + model.key;
  checkbox.checked = true;
  checkbox.addEventListener('change', updateDownloads);
  label.append(checkbox, model.name);
  modelOptions.append(label);

  const card = node('article', undefined, 'model');
  card.dataset.model = model.key;
  const head = node('div', undefined, 'model-head');
  const meta = node('p', `${model.quantization} · 4-bit weights with FP32 where applicable · ~${model.downloadMB} MB · `);
  meta.append(link(model.license, model.licenseUrl));
  head.append(
    node('h3', model.name),
    link(model.checkpoint, `https://huggingface.co/${model.checkpoint}/tree/${model.revision}`, 'checkpoint'),
    node('code', model.revision),
    meta,
  );
  const body = node('div', undefined, 'model-body');
  body.append(node('p', 'Ready to run.'));
  modelBodies.set(model.key, body);
  card.append(head, body);
  document.querySelector('#results').append(card);
}

function renderReply(entry, backend, displayName, openRaw) {
  const reply = node('section', undefined, 'reply');
  const result = entry.result;
  reply.dataset.backend = backend;
  reply.append(node('h4', result?.backendUsed ?? displayName));
  if (!result) {
    const isProblem = entry.state === 'unavailable' || entry.state === 'failed';
    reply.append(node('p', entry.message || 'Queued.', isProblem ? 'error' : 'meta'));
    return reply;
  }

  reply.append(node('p', result.text || '(No visible text generated.)', 'answer'));
  const dispatchText = result.gpuDispatches > 0 ? ` · ${result.gpuDispatches.toLocaleString()} GPU dispatches` : '';
  reply.append(node('p', `${result.tokenIds.length} tokens · ${(result.elapsedMs / 1000).toFixed(2)} s · ${result.stop}${dispatchText}`, 'meta'));
  const raw = node('details');
  raw.open = openRaw;
  raw.append(node('summary', 'Raw completion'), node('pre', result.raw));
  reply.append(raw);
  return reply;
}

function renderModel(model) {
  const body = modelBodies.get(model.key);
  const modelRun = run.models.find((item) => item.key === model.key);
  const openDetails = new Set([...body.querySelectorAll('details[open]')].map((detail) => detail.parentElement.dataset.backend));

  if (!modelRun) {
    body.replaceChildren(node('p', 'Skipped for this run.'));
    return;
  }

  const shownIndex = Number(promptSelect.value || 0);
  const entries = run.backends.map((backend) =>
    renderReply(modelRun.responses[backend][shownIndex], backend, backendNames[backend], openDetails.has(backend)));
  body.replaceChildren(...entries);
}

function renderPrompt() {
  shownPrompt.textContent = run.prompts[Number(promptSelect.value || 0)];
  for (const model of MODELS) renderModel(model);
}

function executeSession(model, backend) {
  const entries = model.responses[backend];
  return new Promise((resolve) => {
    let worker;
    let index = 0;
    let settled = false;
    function finish(state, message) {
      if (settled) return;
      settled = true;
      worker?.terminate();
      cancelSession = null;
      if (state) {
        for (const entry of entries) {
          if (!entry.result) Object.assign(entry, { state, message });
        }
      }
      renderModel(model);
      resolve();
    }

    try {
      worker = new Worker(new URL('./inference.js', import.meta.url), { type: 'module' });
      worker.addEventListener('error', (event) => finish('failed', event.message || 'The inference worker stopped unexpectedly.'));
      worker.addEventListener('messageerror', () => finish('failed', 'The browser could not read the inference worker’s response.'));
      worker.addEventListener('message', ({ data }) => {
        if (settled) return;
        switch (data.type) {
          case 'download':
            if (index === 0 && !entries[0].result) {
              entries[0].message = `Loading model files · ${Math.round(data.bytes / 1_000_000)} MB read`;
              if (Number(promptSelect.value) === 0) renderModel(model);
            }
            break;
          case 'generating':
            index = data.index;
            entries[index].message = 'Generating…';
            status.textContent = `${model.name} · ${backendNames[backend]} · prompt ${index + 1} of ${entries.length}`;
            if (Number(promptSelect.value) === index) renderModel(model);
            break;
          case 'result':
            entries[data.index] = { state: 'complete', result: data.result };
            if (Number(promptSelect.value) === data.index) renderModel(model);
            break;
          case 'failed':
            for (const entry of entries) {
              if (!entry.result) Object.assign(entry, { state: 'failed', message: data.message });
            }
            renderModel(model);
            break;
          case 'done':
            finish();
            break;
        }
      });
      cancelSession = () => finish('stopped', 'Stopped.');
      entries[0].message = 'Preparing runtime and model files…';
      status.textContent = `Loading ${model.name} · ${backendNames[backend]}…`;
      renderModel(model);
      worker.postMessage({ key: model.key, backend, prompts: run.prompts });
    } catch (error) {
      finish('failed', error.message);
    }
  });
}

async function startRun(prompts) {
  if (controls.disabled) return;
  const picked = selectedModels();
  if (!picked.length) return;
  const backends = backendInput.value === 'both' ? ['webgpu', 'wasm'] : [backendInput.value];
  if (backendInput.value === 'wasm' && picked.every((model) => model.wasmUnavailable)) return;
  run = {
    formatVersion: 1,
    startedAt: new Date().toISOString(),
    runtime: { transformersJs: TRANSFORMERS_VERSION, onnxRuntimeWeb: ORT_VERSION },
    browser: navigator.userAgent,
    generation: {
      do_sample: false,
      num_beams: 1,
      max_new_tokens: MAX_NEW_TOKENS,
      context: 'One user message per inference, rendered by the checkpoint tokenizer’s bundled chat template.',
    },
    prompts,
    backends,
    models: picked.map((model) => ({
      ...model,
      responses: Object.fromEntries(backends.map((backend) => [
        backend, prompts.map(() => backend === 'wasm' && model.wasmUnavailable
          ? { state: 'unavailable', message: model.wasmUnavailable }
          : { state: 'queued' }),
      ])),
    })),
    stopped: false,
  };
  controls.disabled = true;
  stopButton.hidden = false;
  exportButton.disabled = true;
  promptSelect.replaceChildren(...prompts.map((text, index) => new Option(text, index)));
  document.querySelector('#view-controls').hidden = prompts.length === 1;
  shownPrompt.hidden = prompts.length > 1;
  renderPrompt();

  try {
    for (const model of run.models) {
      for (const backend of backends) {
        if (!run.stopped && model.responses[backend][0].state === 'queued') {
          await executeSession(model, backend);
        }
      }
    }
    const allEntries = run.models.flatMap((model) => Object.values(model.responses).flat());
    for (const entry of allEntries) {
      if (entry.state === 'queued') Object.assign(entry, { state: 'stopped', message: 'Stopped.' });
    }
    const complete = allEntries.filter((entry) => entry.state === 'complete').length;
    const unavailable = allEntries.filter((entry) => entry.state === 'unavailable').length;
    const unverified = allEntries.filter((entry) => entry.result?.backendUsed.includes('unverified')).length;
    status.textContent = `${run.stopped ? 'Stopped' : 'Finished'} · ${complete} of ${allEntries.length - unavailable} responses generated.` +
      (unavailable ? ` ${unavailable} unavailable on WASM.` : '') +
      (unverified ? ` GPU compute was unverified for ${unverified}.` : '');
    renderPrompt();
  } finally {
    run.finishedAt = new Date().toISOString();
    controls.disabled = false;
    stopButton.hidden = true;
    exportButton.disabled = false;
  }
}

form.addEventListener('submit', (event) => {
  event.preventDefault();
  if (!promptInput.value.trim()) {
    promptInput.setCustomValidity('Enter a prompt.');
    promptInput.reportValidity();
    return;
  }
  void startRun([promptInput.value]);
});
promptInput.addEventListener('input', () => promptInput.setCustomValidity(''));
backendInput.addEventListener('change', updateDownloads);
checksButton.addEventListener('click', () => void startRun(CHECKS.map(({ prompt }) => prompt)));
promptSelect.addEventListener('change', renderPrompt);
stopButton.addEventListener('click', () => {
  run.stopped = true;
  cancelSession?.();
});
exportButton.addEventListener('click', () => {
  const objectUrl = URL.createObjectURL(new Blob([JSON.stringify(run, null, 2)], { type: 'application/json' }));
  const download = node('a');
  download.href = objectUrl;
  download.download = `browser-models-${run.startedAt.replaceAll(':', '-')}.json`;
  document.body.append(download);
  download.click();
  download.remove();
  setTimeout(() => URL.revokeObjectURL(objectUrl), 1_000);
});

promptInput.maxLength = MAX_PROMPT_LENGTH;
document.querySelector('#runtime').textContent =
  `Transformers.js ${TRANSFORMERS_VERSION} · ONNX Runtime Web ${ORT_VERSION}`;
updateDownloads();
showCapability();
