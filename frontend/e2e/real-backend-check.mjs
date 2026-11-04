// SOLO PRUEBAS. Recorre la app contra un backend real (con el proxy de desarrollo o en el pod).
// Uso: node e2e/real-backend-check.mjs http://localhost:4200
// No hace suposiciones sobre las cifras: solo comprueba que lo que llega se dibuja.
import { mkdirSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const BASE = process.argv[2] ?? 'http://localhost:4200';
const here = dirname(fileURLToPath(import.meta.url));
const FIX = resolve(here, '.fixtures');
const SHOTS = resolve(here, '../.screenshots');
mkdirSync(SHOTS, { recursive: true });

const results = [];
const check = (name, ok, detail = '') => {
  results.push(ok);
  console.log(`${ok ? 'OK  ' : 'FALLA'} ${name}${detail ? `: ${detail}` : ''}`);
};

const browser = await chromium.launch({ channel: 'chrome' });
const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
const page = await context.newPage();
const q = (id) => page.locator(`[data-testid="${id}"]`);
page.on('pageerror', (err) => check('sin excepciones', false, err.message));

const config = await (await context.request.get(`${BASE}/api/v1/config`)).json();
check('GET /config por el proxy', Array.isArray(config.classes), `${config.classes?.length} clases, ${config.moods?.length} moods`);

await page.goto(`${BASE}/`);
await q('dropzone').waitFor();
check('los límites vienen de /config', (await q('dropzone-limits').textContent()).includes(`${config.limits.min_side_px} px`));

// Validación local y del servidor
await q('file-input').setInputFiles(join(FIX, 'not-an-image.jpg'));
await page.waitForSelector('[data-error-code="UNSUPPORTED_FORMAT"]');
check('archivo inválido rechazado', true);
const raw = await context.request.post(`${BASE}/api/v1/animate`, {
  multipart: { file: { name: 'x.txt', mimeType: 'text/plain', buffer: Buffer.from('hola, no soy imagen') } },
});
const rawBody = await raw.json();
check('el servidor usa el formato único de error', rawBody.error?.code === 'UNSUPPORTED_FORMAT', `${raw.status()} ${rawBody.error?.code}`);

// Rayos X con la respuesta real
const analyzed = page.waitForResponse((r) => r.url().endsWith('/api/v1/analyze'), { timeout: 60000 });
await q('file-input').setInputFiles(join(FIX, 'lake.png'));
const analyzeBody = await (await analyzed).json();
await page.waitForSelector('[data-testid="xray-section"] [data-testid="inference-total"]', { timeout: 60000 });
console.log('     /analyze:', JSON.stringify({ ...analyzeBody, mask_png_base64: `<${analyzeBody.mask_png_base64?.length} chars>` }));
const labelsOk = await Promise.all(analyzeBody.classes.map(async (c, i) => {
  const expected = config.classes.find((k) => k.id === c.id)?.label;
  return expected === undefined || (await q('class-label').nth(i).textContent()).trim() === expected;
}));
check('/config trae label en todas las clases', config.classes.every((c) => typeof c.label === 'string' && c.label.length > 0));
check('las etiquetas mostradas son las de /config', labelsOk.every(Boolean));
const rows = await q('class-row').count();
check('una fila por clase recibida', rows === analyzeBody.classes.length, `${rows}/${analyzeBody.classes.length}`);
check('modelo mostrado = recibido', (await q('model-version').textContent()).includes(analyzeBody.model_version));
if (analyzeBody.model_version === 'dummy-dev') check('aviso de dummy-dev visible', await q('dummy-warning').isVisible());
await page.waitForTimeout(1800);
const painted = await page.locator('app-xray-overlay canvas').evaluate((canvas) => {
  const { data, width, height } = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height);
  const colors = new Set();
  for (let i = 0; i < data.length; i += 4 * 97) if (data[i + 3]) colors.add(`${data[i]},${data[i + 1]},${data[i + 2]}`);
  return { width, height, colors: [...colors] };
});
check('máscara real decodificada y pintada', painted.colors.length > 0, JSON.stringify(painted));
check('la máscara mide lo que dice la API', painted.width === analyzeBody.width && painted.height === analyzeBody.height);
const allowed = new Set(config.classes.map((c) => [1, 3, 5].map((s) => parseInt(c.color.slice(s, s + 2), 16)).join(',')));
check('solo colores de /config', painted.colors.every((c) => allowed.has(c)));
await page.screenshot({ path: join(SHOTS, 'real-1440-rayos-x.png') });

// Pincel y animación
await q('tab-brush').click();
const canvas = q('brush-canvas');
const box = await canvas.boundingBox();
await page.mouse.move(box.x + box.width * 0.2, box.y + box.height * 0.3);
await page.mouse.down();
await page.mouse.move(box.x + box.width * 0.6, box.y + box.height * 0.35, { steps: 12 });
await page.mouse.up();

const responses = [];
const jobUrls = [];
page.on('response', (r) => {
  if (r.url().includes('/api/v1/animate')) responses.push(r.status());
  if (r.url().includes('/api/v1/jobs/')) jobUrls.push(r.url());
});
const lastJobUrl = async () => jobUrls[jobUrls.length - 1];
const started = Date.now();
await q('animate').click();
// La espera debe verse viva con datos reales
await page.waitForSelector('[data-phase="running"]', { timeout: 60000 });
await page.waitForFunction(() => /[3-9] s|\d\d s/.test(document.querySelector('[data-testid="job-elapsed"]')?.textContent ?? ''), null, { timeout: 30000 }).catch(() => undefined);
check('progreso: cronómetro en marcha', /[1-9]\d* s/.test((await q('job-elapsed').textContent().catch(() => '')) ?? ''));
check('progreso: etapa actual marcada', (await page.locator('[data-testid="job-step"][data-state="current"]').count()) === 1, await page.locator('[data-testid="job-step"][data-state="current"]').textContent().catch(() => ''));
await page.screenshot({ path: join(SHOTS, 'real-1440-progreso.png') });
const outcome = await Promise.race([
  page.waitForSelector('[data-testid="comparator-handle"]', { timeout: 280000 }).then(() => 'ok'),
  page.waitForSelector('[data-testid="job-error"]', { timeout: 280000 }).then(() => 'error'),
]);
check('POST /animate con pincel aceptado', responses.includes(202), responses.join(','));
if (outcome === 'ok') {
  check('job real terminado', true, `${Date.now() - started} ms`);
  const job = await (await context.request.get(await lastJobUrl())).json();
  console.log('     job:', JSON.stringify({ seed: job.seed, stage: job.stage, timings_ms: job.timings_ms, result: job.result }));
  check('semilla mostrada = job', (await q('result-seed').textContent()).trim() === String(job.seed), String(job.seed));
  for (const [key, value] of Object.entries(job.timings_ms)) {
    const shown = (await q(`timing-${key}`).textContent()).trim();
    check(`tiempo ${key}`, value === null ? shown === 'sin medir' : /d (ms|s)$/.test(shown), `${value} -> ${shown}`);
  }
  await page.waitForTimeout(2500);
  const video = await page.locator('[data-testid="after-layer"] video').evaluate((v) => ({ readyState: v.readyState, w: v.videoWidth, h: v.videoHeight, d: v.duration, err: v.error?.message }));
  check('el MP4 real se reproduce en el navegador', video.readyState >= 2 && video.w > 0, JSON.stringify(video));
  await page.screenshot({ path: join(SHOTS, 'real-1440-comparador.png') });

  await q('export-other').click();
  await Promise.race([
    page.waitForFunction(() => document.querySelector('[data-testid="after-layer"] img')?.naturalWidth > 0, null, { timeout: 280000 }),
    page.waitForSelector('[data-testid="job-error"]', { timeout: 280000 }),
  ]);
  const gif = await page.locator('[data-testid="after-layer"] img').count();
  check('el GIF real se muestra', gif === 1);
  await page.waitForTimeout(1200);
  await page.screenshot({ path: join(SHOTS, 'real-1440-gif.png') });
} else {
  const code = await page.locator('[data-error-code]').getAttribute('data-error-code');
  check('job real terminado', false, `${code}: ${await page.locator('[data-error-code]').textContent()}`);
}

// Métricas
await page.goto(`${BASE}/modelo`);
await Promise.race([q('metrics-tiles').waitFor(), q('metrics-error').waitFor()]);
const metricsResponse = await context.request.get(`${BASE}/api/v1/model/metrics`);
if (metricsResponse.ok()) {
  const metrics = await metricsResponse.json();
  check('mIoU mostrado = respuesta', (await q('metric-miou').textContent()).trim() === (metrics.miou === null ? 'sin dato' : String(metrics.miou)), String(metrics.miou));
  if (metrics.dummy || metrics.split === 'dummy') check('aviso de métricas dummy', await q('metrics-dummy').isVisible());
  const cells = await q('confusion-cell').count();
  check('matriz completa', cells === metrics.confusion_matrix.flat().length, String(cells));
} else {
  check('sin métricas se muestra el error', await q('metrics-error').isVisible());
}
await page.screenshot({ path: join(SHOTS, 'real-1440-metricas.png') });

// Expo con consentimiento, y galería
const phone = await browser.newContext({ viewport: { width: 390, height: 844 }, hasTouch: true, isMobile: true });
const mobile = await phone.newPage();
await mobile.goto(`${BASE}/expo/subir`);
await mobile.locator('[data-testid="expo-gallery"]').setInputFiles(join(FIX, 'forest.png'));
await mobile.waitForSelector('[data-testid="expo-preview"]');
await mobile.locator('[data-testid="consent"]').check();
await mobile.click('[data-testid="expo-animate"]');
const expoOutcome = await Promise.race([
  mobile.waitForSelector('[data-testid="expo-result"]', { timeout: 280000 }).then(() => 'ok'),
  mobile.waitForSelector('[data-testid="job-error"]', { timeout: 280000 }).then(() => 'error'),
]);
check('expo: animación con consentimiento', expoOutcome === 'ok');
await mobile.screenshot({ path: join(SHOTS, 'real-390-expo.png') });
await phone.close();

await page.goto(`${BASE}/expo/pantalla`);
await q('wall-qr').waitFor();
if (expoOutcome === 'ok') {
  const seen = await page.waitForFunction(() => document.querySelectorAll('[data-testid="wall-tile"] video').length > 0, null, { timeout: 15000 }).then(() => true, () => false);
  check('expo: aparece en el muro', seen);
  await page.waitForTimeout(1500);
}
check('muro: QR apunta a public_base_url', (await q('wall-url').textContent()).includes(`${config.public_base_url.replace(/\/+$/, '')}/expo/subir`), await q('wall-url').textContent());
await page.screenshot({ path: join(SHOTS, 'real-1440-muro.png') });

await browser.close();
const failed = results.filter((ok) => !ok).length;
console.log(`\n${results.length - failed}/${results.length} comprobaciones correctas contra ${BASE}`);
process.exitCode = failed ? 1 : 0;
