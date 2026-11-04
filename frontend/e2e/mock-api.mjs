// SOLO PRUEBAS. API de prueba que cumple el contrato 1.2.0 y sirve el build de producción
// en el mismo origen, como hará Nginx. Todas sus cifras son datos de prueba inventados.
// Uso: node e2e/mock-api.mjs [puerto]. Se controla con GET /__mock/set?clave=valor.
import { createReadStream, existsSync, readFileSync, statSync } from 'node:fs';
import { createServer } from 'node:http';
import { dirname, extname, join, normalize, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { PNG } from 'pngjs';
import { renderMask } from './fixtures.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const DIST = resolve(here, '../dist/frontend/browser');
const FIXTURES = resolve(here, '.fixtures');
const API = '/api/v1';

const CONFIG = {
  public_base_url: 'http://192.168.0.10:8080',
  limits: { max_upload_bytes: 15728640, max_video_seconds: 10, min_side_px: 256, formats: ['jpg', 'png', 'webp', 'mp4', 'mov'] },
  moods: ['calma', 'epico', 'magico', 'nostalgico'],
  classes: [
    { id: 0, name: 'fondo', label: 'Fondo', color: '#3A3F4B' },
    { id: 1, name: 'cielo', label: 'Cielo', color: '#4FC3F7' },
    { id: 2, name: 'agua', label: 'Agua', color: '#2979FF' },
    { id: 3, name: 'vegetacion', label: 'Vegetación', color: '#66BB6A' },
    { id: 4, name: 'fuego_luz', label: 'Fuego / luz', color: '#FF7043' },
    { id: 5, name: 'persona', label: 'Persona', color: '#EC407A' },
    { id: 6, name: 'animal', label: 'Animal', color: '#FFCA28' },
  ],
};

const FAKE_METRICS_NAMES = CONFIG.classes.map((c) => c.name);
const FAKE_METRICS = `{
  "model_sha256": "${'ab'.repeat(32)}",
  "dataset_manifest_sha256": "${'cd'.repeat(32)}",
  "split": "test",
  "num_images": 111,
  "evaluated_at": "2026-01-01T00:00:00Z",
  "miou": 0.1234,
  "dice_macro": 0.2345,
  "pixel_accuracy": 0.3456,
  "per_class": {
${FAKE_METRICS_NAMES.map((n, i) => `    "${n}": { "iou": 0.1${i}, "dice": 0.2${i}, "support_pixels": ${(i + 1) * 123456} }`).join(',\n')}
  },
  "confusion_matrix": ${JSON.stringify(FAKE_METRICS_NAMES.map((_, r) => FAKE_METRICS_NAMES.map((_, c) => (r === c ? 900000 - r * 70000 : 8000 * ((r + c) % 5 + 1)))))},
  "classes": ${JSON.stringify(FAKE_METRICS_NAMES)}
}`;

const state = {
  modelVersion: 'fixture-prueba',
  queueFull: false,
  metricsMissing: false,
  metricsDummy: false,
  apiDown: false,
  gallery: 0,
  jobPolls: 4,
  jobFailCode: '',
  analyzeFailCode: '',
  jobs: new Map(),
  lastAnimate: null,
  lastAnalyzeBytes: 0,
};

const ERRORS = {
  MODEL_NOT_READY: [503, 'El modelo aún no está cargado.'],
  INTERNAL_ERROR: [500, 'Ocurrió un error interno.'],
  CORRUPT_FILE: [400, 'No se pudo decodificar el archivo.'],
};

function send(res, status, body, headers = {}) {
  const payload = typeof body === 'string' ? body : JSON.stringify(body);
  res.writeHead(status, { 'Content-Type': 'application/json', ...headers });
  res.end(payload);
}

function sendError(res, status, code, message, headers) {
  send(res, status, { error: { code, message, details: null } }, headers);
}

function readBody(req) {
  return new Promise((done) => {
    const chunks = [];
    req.on('data', (c) => chunks.push(c));
    req.on('end', () => done(Buffer.concat(chunks)));
  });
}

function parseMultipart(req, body) {
  const boundary = `--${/boundary=(.+)$/.exec(req.headers['content-type'] ?? '')?.[1]}`;
  const fields = {};
  let file = null;
  let cursor = body.indexOf(boundary);
  while (cursor !== -1) {
    const next = body.indexOf(boundary, cursor + boundary.length);
    if (next === -1) break;
    const part = body.subarray(cursor + boundary.length + 2, next - 2);
    const split = part.indexOf('\r\n\r\n');
    const head = part.subarray(0, split).toString();
    const value = part.subarray(split + 4);
    const name = /name="([^"]+)"/.exec(head)?.[1];
    if (/filename=/.test(head)) file = value;
    else if (name) fields[name] = value.toString();
    cursor = next;
  }
  return { fields, file };
}

function pngSize(file) {
  try {
    if (file && file[0] === 0x89) {
      const png = PNG.sync.read(file);
      return [png.width, png.height];
    }
  } catch {
    /* Se usa el tamaño por defecto. */
  }
  return [1280, 720];
}

function analyzeResponse(file) {
  const [w, h] = pngSize(file);
  const scale = Math.min(1, 1280 / Math.max(w, h));
  const width = Math.round(w * scale);
  const height = Math.round(h * scale);
  const { buffer, counts } = renderMask(width, height);
  const total = width * height;
  const classes = [...counts.entries()]
    .map(([id, count]) => ({
      id,
      name: CONFIG.classes[id].name,
      pixel_pct: Math.round((count / total) * 1000) / 10,
      confidence: Math.round((0.61 + (id * 37 % 31) / 100) * 100) / 100,
    }))
    .sort((a, b) => b.pixel_pct - a.pixel_pct);
  return {
    width,
    height,
    mask_png_base64: buffer.toString('base64'),
    classes,
    inference_ms: { segmentation: 12.3, depth: null, total: 45.6 },
    depth_png_base64: null,
    model_version: state.modelVersion,
    model_sha256: 'ef'.repeat(32),
    conf_threshold: 0.5,
  };
}

function jobView(job) {
  job.polls += 1;
  const stages = ['queued', 'segmenting', 'depth', 'rendering', 'encoding'];
  const fraction = Math.min(1, job.polls / state.jobPolls);
  const base = {
    job_id: job.id,
    mood: job.mood,
    seed: 12345,
    created_at: '2026-01-01T00:00:00Z',
    updated_at: '2026-01-01T00:00:00Z',
    result: null,
    timings_ms: null,
    error: null,
  };
  if (fraction >= 1) {
    if (state.jobFailCode) {
      return { ...base, status: 'failed', stage: 'rendering', progress: 0.5, error: { code: state.jobFailCode, message: 'El trabajo se interrumpió por un reinicio del servidor.' } };
    }
    return {
      ...base,
      status: 'succeeded',
      stage: 'done',
      progress: 1,
      result: { url: `${API}/results/${job.id}.${job.format}`, format: job.format, width: 1280, height: 720, duration_s: 4, fps: 24, size_bytes: 1234567 },
      timings_ms: { segmentation: 11.1, depth: null, render: 33.3, encode: 44.4, total: 8910.5 },
    };
  }
  const stage = stages[Math.min(stages.length - 1, Math.floor(fraction * stages.length))];
  return { ...base, status: stage === 'queued' ? 'queued' : 'running', stage, progress: stage === 'queued' ? 0 : fraction };
}

function serveFile(req, res, path, type) {
  const size = statSync(path).size;
  const range = /bytes=(\d*)-(\d*)/.exec(req.headers.range ?? '');
  if (range) {
    const start = Number(range[1] || 0);
    const end = Number(range[2] || size - 1);
    res.writeHead(206, { 'Content-Type': type, 'Content-Range': `bytes ${start}-${end}/${size}`, 'Accept-Ranges': 'bytes', 'Content-Length': end - start + 1 });
    createReadStream(path, { start, end }).pipe(res);
  } else {
    res.writeHead(200, { 'Content-Type': type, 'Content-Length': size, 'Accept-Ranges': 'bytes' });
    createReadStream(path).pipe(res);
  }
}

const MIME = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml', '.woff2': 'font/woff2', '.woff': 'font/woff' };

function serveStatic(req, res, pathname) {
  const candidate = normalize(join(DIST, pathname));
  const target = candidate.startsWith(DIST) && existsSync(candidate) && statSync(candidate).isFile() ? candidate : join(DIST, 'index.html');
  res.writeHead(200, { 'Content-Type': MIME[extname(target)] ?? 'application/octet-stream' });
  res.end(readFileSync(target));
}

const server = createServer(async (req, res) => {
  const url = new URL(req.url, 'http://mock');
  const path = url.pathname;

  if (path === '/__mock/set') {
    for (const [key, value] of url.searchParams) {
      state[key] = value === 'true' ? true : value === 'false' ? false : /^\d+$/.test(value) ? Number(value) : value;
    }
    return send(res, 200, { ok: true });
  }
  if (path === '/__mock/last') return send(res, 200, { lastAnimate: state.lastAnimate });
  if (!path.startsWith('/api/')) return serveStatic(req, res, path);
  if (state.apiDown) {
    res.writeHead(502, { 'Content-Type': 'text/html' });
    return res.end('<html>Bad gateway</html>');
  }

  if (path === `${API}/config`) return send(res, 200, CONFIG);
  if (path === `${API}/health`) {
    return send(res, 200, { status: 'ok', model_loaded: true, depth_loaded: true, model_version: state.modelVersion, model_sha256: 'ef'.repeat(32), depth_sha256: '12'.repeat(32) });
  }
  if (path === `${API}/model/metrics`) {
    if (state.metricsMissing) return sendError(res, 404, 'METRICS_NOT_FOUND', 'Falta metrics.json en el servidor.');
    if (state.metricsDummy) {
      return send(res, 200, FAKE_METRICS.replace('"split": "test"', '"dummy": true,\n  "split": "dummy"').replace('"miou": 0.1234', '"miou": null').replace('"dice_macro": 0.2345', '"dice_macro": null').replace('"pixel_accuracy": 0.3456', '"pixel_accuracy": null'));
    }
    return send(res, 200, FAKE_METRICS);
  }
  if (path === `${API}/analyze` && req.method === 'POST') {
    const { file } = parseMultipart(req, await readBody(req));
    if (state.analyzeFailCode) {
      const [status, message] = ERRORS[state.analyzeFailCode] ?? [500, 'Error de prueba.'];
      return sendError(res, status, state.analyzeFailCode, message);
    }
    return send(res, 200, analyzeResponse(file));
  }
  if (path === `${API}/animate` && req.method === 'POST') {
    const { fields, file } = parseMultipart(req, await readBody(req));
    state.lastAnimate = { fields, fileBytes: file?.length ?? 0 };
    if (state.queueFull) {
      return sendError(res, 429, 'QUEUE_FULL', 'Hay 8 trabajos en cola; intenta de nuevo en un momento.', { 'Retry-After': '30' });
    }
    const id = `00000000-0000-4000-8000-${String(state.jobs.size + 1).padStart(12, '0')}`;
    state.jobs.set(id, { id, polls: 0, mood: fields.mood ?? 'calma', format: fields.output_format ?? 'mp4' });
    return send(res, 202, { job_id: id, status: 'queued', queue_position: 2, status_url: `${API}/jobs/${id}` });
  }
  const jobMatch = new RegExp(`^${API}/jobs/(.+)$`).exec(path);
  if (jobMatch) {
    const job = state.jobs.get(jobMatch[1]);
    return job ? send(res, 200, jobView(job)) : sendError(res, 404, 'JOB_NOT_FOUND', 'El job no existe.');
  }
  const resultMatch = new RegExp(`^${API}/results/(.+)\\.(mp4|gif)$`).exec(path);
  if (resultMatch) {
    const variants = ['result-lake.mp4', 'result-forest.mp4', 'result-night.mp4'];
    const n = parseInt(resultMatch[1].slice(-2), 16) || 0;
    return serveFile(req, res, join(FIXTURES, variants[n % variants.length]), 'video/mp4');
  }
  if (path === `${API}/gallery`) {
    const limit = Number(url.searchParams.get('limit') ?? 24);
    const items = Array.from({ length: Math.min(state.gallery, limit) }, (_, i) => ({
      job_id: `aaaaaaaa-0000-4000-8000-${i.toString(16).padStart(12, '0')}`,
      mood: CONFIG.moods[i % CONFIG.moods.length],
      result_url: `${API}/results/aaaaaaaa-0000-4000-8000-${i.toString(16).padStart(12, '0')}.mp4`,
      created_at: new Date(Date.UTC(2026, 0, 1, 12, 0, 60 - i)).toISOString(),
    }));
    return send(res, 200, { items });
  }
  return sendError(res, 404, 'NOT_FOUND', 'Ruta de prueba inexistente.');
});

const port = Number(process.argv[2] ?? 0);
server.listen(port, '127.0.0.1', () => {
  console.log(`MOCK_PORT=${server.address().port}`);
});
