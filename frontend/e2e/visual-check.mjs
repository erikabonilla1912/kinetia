// SOLO PRUEBAS. Recorre la app real en Chrome sin interfaz contra la API de prueba.
// Toma capturas a 390, 1440 y 1920 px y comprueba scroll horizontal, área de toque,
// contraste AA (axe-core), QR legible y la forma de lo que se envía a la API.
// Requiere `npm run build`. Uso: npm run verify:visual
import { spawn } from 'node:child_process';
import { mkdirSync, readFileSync, readdirSync, rmSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import jsQR from 'jsqr';
import { chromium } from 'playwright';
import { PNG } from 'pngjs';

const require = createRequire(import.meta.url);
const axeSource = readFileSync(require.resolve('axe-core/axe.min.js'), 'utf8');

const here = dirname(fileURLToPath(import.meta.url));
const SHOTS = resolve(here, '../.screenshots');
const FIX = resolve(here, '.fixtures');
mkdirSync(SHOTS, { recursive: true });
// Las capturas real-* son de la prueba contra la API real: se conservan.
for (const file of readdirSync(SHOTS)) if (!file.startsWith('real-')) rmSync(join(SHOTS, file));

const VIEWPORTS = [
  { name: '390', width: 390, height: 844, hasTouch: true, isMobile: true, deviceScaleFactor: 2 },
  { name: '1440', width: 1440, height: 900 },
  { name: '1920', width: 1920, height: 1080 },
];

const issues = [];
const notes = [];

const mock = spawn(process.execPath, [join(here, 'mock-api.mjs'), '0'], { stdio: ['ignore', 'pipe', 'inherit'] });
const port = await new Promise((done) => {
  mock.stdout.on('data', (d) => {
    const m = /MOCK_PORT=(\d+)/.exec(String(d));
    if (m) done(Number(m[1]));
  });
});
const BASE = `http://127.0.0.1:${port}`;
const set = (params) => fetch(`${BASE}/__mock/set?${new URLSearchParams(params)}`);
const defaults = { modelVersion: 'fixture-prueba', queueFull: false, metricsMissing: false, metricsDummy: false, apiDown: false, gallery: 0, jobPolls: 4, jobFailCode: '', analyzeFailCode: '' };

const browser = await chromium.launch({ channel: 'chrome' });

async function audit(page, viewport, label) {
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  if (overflow > 0) issues.push(`[${viewport.name}] ${label}: scroll horizontal de ${overflow}px`);

  const small = await page.evaluate(() => {
    const out = [];
    const nodes = document.querySelectorAll('button, a[href], [role="slider"], input[type="range"], label:has(input), summary');
    for (const el of nodes) {
      if (el.closest('[aria-hidden="true"]')) continue;
      const r = el.getBoundingClientRect();
      if (r.width === 0 || r.height === 0) continue;
      if (r.height < 43.5 || (r.width < 43.5 && !el.matches('summary'))) {
        out.push(`${el.tagName.toLowerCase()}("${(el.textContent || el.getAttribute('aria-label') || '').trim().slice(0, 24)}") ${Math.round(r.width)}x${Math.round(r.height)}`);
      }
    }
    return out;
  });
  if (small.length) issues.push(`[${viewport.name}] ${label}: toque <44px: ${small.join('; ')}`);

  await page.evaluate(axeSource);
  const violations = await page.evaluate(async () => {
    const r = await window.axe.run(document, { runOnly: ['wcag2a', 'wcag2aa'] });
    return r.violations.map((v) => `${v.id}: ${v.nodes.slice(0, 3).map((n) => n.target.join(' ')).join(' | ')}`);
  });
  if (violations.length) issues.push(`[${viewport.name}] ${label}: axe ${violations.join(' ; ')}`);
}

async function shot(page, viewport, label, { fullPage = viewport.name === '390', check = true } = {}) {
  await page.waitForTimeout(150);
  await page.screenshot({ path: join(SHOTS, `${viewport.name}-${label}.png`), fullPage });
  if (check) await audit(page, viewport, label);
}

async function upload(page, file) {
  await page.setInputFiles('[data-testid="file-input"]', join(FIX, file));
}

function checkPayload(raw, viewport) {
  const payload = JSON.parse(raw);
  let total = 0;
  for (const s of payload.strokes) {
    total += s.points.length;
    if (s.radius < 0.005 || s.radius > 0.2) issues.push(`[${viewport.name}] radio fuera de rango ${s.radius}`);
    if (s.direction_deg < 0 || s.direction_deg > 360) issues.push(`[${viewport.name}] dirección fuera de rango`);
    if (s.strength < 0 || s.strength > 1) issues.push(`[${viewport.name}] fuerza fuera de rango`);
    for (const [x, y] of s.points) if (x < 0 || x > 1 || y < 0 || y > 1) issues.push(`[${viewport.name}] punto fuera de rango`);
  }
  if (payload.strokes.length > 200 || total > 2000) issues.push(`[${viewport.name}] límites de pincel excedidos`);
  return payload;
}

for (const viewport of VIEWPORTS) {
  const { name, ...options } = viewport;
  const context = await browser.newContext({ viewport: { width: options.width, height: options.height }, ...options });
  const page = await context.newPage();
  page.on('pageerror', (err) => issues.push(`[${name}] excepción: ${err.message}`));
  page.on('console', (msg) => {
    if (msg.type() === 'error' && !/status of (4|5)\d\d/.test(msg.text()) && !/ERR_/.test(msg.text())) {
      issues.push(`[${name}] consola: ${msg.text()}`);
    }
  });
  await set(defaults);

  // Portada y validaciones
  await page.goto(`${BASE}/`);
  await page.waitForSelector('[data-testid="dropzone"]');
  await page.waitForTimeout(1300);
  await shot(page, viewport, '01-portada');
  await upload(page, 'not-an-image.jpg');
  await page.waitForSelector('[data-error-code="UNSUPPORTED_FORMAT"]');
  await shot(page, viewport, '02-archivo-invalido');
  await upload(page, 'tiny.png');
  await page.waitForSelector('[data-error-code="RESOLUTION_TOO_LOW"]');

  // Rayos X
  await upload(page, 'lake.png');
  await page.waitForSelector('[data-testid="class-row"]');
  await page.waitForTimeout(1900);
  await shot(page, viewport, '03-rayos-x');
  await page.locator('[data-testid="class-row"]').nth(1).click();
  await page.waitForTimeout(200);
  await shot(page, viewport, '04-rayos-x-una-clase', { check: false });
  await page.locator('[data-testid="class-row"]').nth(1).click();

  // Pincel
  await page.click('[data-testid="tab-brush"]');
  const canvas = page.locator('[data-testid="brush-canvas"]');
  await canvas.scrollIntoViewIfNeeded();
  const box = await canvas.boundingBox();
  const stroke = async (x0, y0, x1, y1) => {
    await page.mouse.move(box.x + box.width * x0, box.y + box.height * y0);
    await page.mouse.down();
    for (let i = 1; i <= 12; i++) {
      await page.mouse.move(box.x + box.width * (x0 + ((x1 - x0) * i) / 12), box.y + box.height * (y0 + ((y1 - y0) * i) / 12));
    }
    await page.mouse.up();
  };
  await stroke(0.1, 0.8, 0.38, 0.72);
  await stroke(0.5, 0.2, 0.85, 0.15);
  await stroke(0.6, 0.65, 0.6, 0.95);
  await page.waitForTimeout(150);
  await shot(page, viewport, '05-pincel');

  // Job: progreso y comparador
  await set({ jobPolls: 6 });
  await page.click('[data-testid="animate"]');
  await page.waitForSelector('[data-phase="running"]');
  await page.waitForTimeout(700);
  await shot(page, viewport, '06-job-en-curso');
  await page.waitForSelector('[data-testid="comparator-handle"]', { timeout: 15000 });
  const last = await (await fetch(`${BASE}/__mock/last`)).json();
  const payload = checkPayload(last.lastAnimate.fields.mask_strokes, viewport);
  if (payload.strokes.length !== 3) issues.push(`[${name}] se esperaban 3 trazos, llegaron ${payload.strokes.length}`);
  const up = payload.strokes[0].direction_deg;
  if (!(up > 270 && up < 360)) issues.push(`[${name}] dirección del primer trazo inesperada: ${up}`);
  const down = payload.strokes[2].direction_deg;
  if (Math.abs(down - 90) > 1) issues.push(`[${name}] dirección del trazo vertical inesperada: ${down}`);
  if (last.lastAnimate.fields.gallery_opt_in !== 'false') issues.push(`[${name}] el estudio no debe publicar en la galería`);
  await page.waitForTimeout(2200);
  await shot(page, viewport, '07-comparador');

  // Comparador con teclado
  const handle = page.locator('[data-testid="comparator-handle"]');
  await handle.focus();
  const keyTo = async (key, expected) => {
    await page.keyboard.press(key);
    await page.waitForFunction((v) => document.querySelector('[data-testid="comparator-handle"]').getAttribute('aria-valuenow') === v, expected, { timeout: 2000 }).catch(() => issues.push(`[${name}] tecla ${key} del comparador`));
  };
  await keyTo('Home', '0');
  await keyTo('End', '100');
  await keyTo('ArrowLeft', '98');

  // Táctil real, por el protocolo del navegador (solo en la vista móvil)
  if (options.hasTouch) {
    const client = await context.newCDPSession(page);
    const swipe = async (x0, y0, x1, y1) => {
      await client.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: [{ x: x0, y: y0 }] });
      for (let i = 1; i <= 8; i++) {
        await client.send('Input.dispatchTouchEvent', { type: 'touchMove', touchPoints: [{ x: x0 + ((x1 - x0) * i) / 8, y: y0 + ((y1 - y0) * i) / 8 }] });
      }
      await client.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] });
    };
    const area = page.locator('[data-testid="comparator-area"]');
    await area.scrollIntoViewIfNeeded();
    const a = await area.boundingBox();
    await swipe(a.x + a.width * 0.7, a.y + a.height * 0.5, a.x + a.width * 0.25, a.y + a.height * 0.5);
    await page.waitForTimeout(100);
    const now = Number(await handle.getAttribute('aria-valuenow'));
    if (Math.abs(now - 25) > 3) issues.push(`[${name}] arrastre táctil del comparador: ${now}`);
    else notes.push(`[${name}] comparador por toque: ${now}`);
    await page.click('[data-testid="tab-brush"]');
    const c = page.locator('[data-testid="brush-canvas"]');
    await c.scrollIntoViewIfNeeded();
    const b = await c.boundingBox();
    const before = await page.locator('[data-testid="brush-usage"]').textContent();
    await swipe(b.x + b.width * 0.2, b.y + b.height * 0.3, b.x + b.width * 0.6, b.y + b.height * 0.3);
    await page.waitForTimeout(100);
    const after = await page.locator('[data-testid="brush-usage"]').textContent();
    if (before === after) issues.push(`[${name}] el pincel no respondió al toque`);
    else notes.push(`[${name}] pincel por toque: "${before.trim()}" -> "${after.trim()}"`);
    await page.click('[data-testid="tab-result"]');
    await page.waitForSelector('[data-testid="comparator-handle"]');
  }
  await page.waitForTimeout(800);
  const videoPlaying = await page.locator('[data-testid="after-layer"] video').evaluate((v) => v.readyState >= 2 && v.videoWidth > 0);
  if (!videoPlaying) issues.push(`[${name}] el video de resultado no se decodificó`);

  // Errores de cola
  await set({ queueFull: true });
  await page.click('[data-testid="animate"]');
  await page.waitForSelector('[data-error-code="QUEUE_FULL"]');
  await page.locator('[data-error-code="QUEUE_FULL"]').scrollIntoViewIfNeeded();
  await shot(page, viewport, '08-cola-llena');
  await set({ queueFull: false, jobFailCode: 'JOB_INTERRUPTED', jobPolls: 1 });

  // Foto vertical
  await page.click('[data-testid="change-file"]');
  await upload(page, 'portrait.png');
  await page.waitForSelector('[data-testid="class-row"]');
  await page.waitForTimeout(1900);
  await shot(page, viewport, '09-foto-vertical');

  // Job fallido
  await page.click('[data-testid="animate"]');
  await page.waitForSelector('[data-error-code="JOB_INTERRUPTED"]');
  await set(defaults);

  // Aviso de modelo dummy
  await set({ modelVersion: 'dummy-dev' });
  await page.goto(`${BASE}/`);
  await page.waitForSelector('[data-testid="dropzone"]');
  await upload(page, 'lake.png');
  await page.waitForSelector('[data-testid="dummy-warning"]');
  await page.waitForTimeout(1800);
  await shot(page, viewport, '10-modelo-dummy');
  await set(defaults);

  // Métricas
  await page.goto(`${BASE}/modelo`);
  await page.waitForSelector('[data-testid="metrics-tiles"]');
  await shot(page, viewport, '11-metricas');
  await set({ metricsDummy: true });
  await page.goto(`${BASE}/modelo`);
  await page.waitForSelector('[data-testid="metrics-dummy"]');
  await shot(page, viewport, '11b-metricas-dummy');
  await set({ metricsDummy: false, metricsMissing: true });
  await page.goto(`${BASE}/modelo`);
  await page.waitForSelector('[data-error-code="METRICS_NOT_FOUND"]');
  await shot(page, viewport, '12-metricas-faltantes');
  await set(defaults);

  // Expo: subir
  await page.goto(`${BASE}/expo/subir`);
  await page.waitForSelector('[data-testid="expo-camera"]', { state: 'attached' });
  const capture = await page.locator('[data-testid="expo-camera"]').evaluate((el) => ({ capture: el.hasAttribute('capture'), accept: el.accept }));
  if (!capture.capture || capture.accept !== 'image/*,video/*') issues.push(`[${name}] input de cámara fuera de contrato`);
  if (await page.locator('[data-testid="consent"]').isChecked()) issues.push(`[${name}] consentimiento marcado por defecto`);
  await shot(page, viewport, '13-expo-subir');
  await page.setInputFiles('[data-testid="expo-gallery"]', join(FIX, 'forest.png'));
  await page.waitForSelector('[data-testid="expo-preview"]');
  await page.locator('[data-testid="consent"]').check();
  await page.waitForTimeout(150);
  await shot(page, viewport, '14-expo-subir-listo');
  await page.click('[data-testid="expo-animate"]');
  await page.waitForSelector('[data-testid="expo-result"]', { timeout: 15000 });
  const expoLast = await (await fetch(`${BASE}/__mock/last`)).json();
  if (expoLast.lastAnimate.fields.gallery_opt_in !== 'true') issues.push(`[${name}] gallery_opt_in no llegó con el consentimiento`);
  await page.waitForTimeout(600);
  await shot(page, viewport, '15-expo-resultado');

  // Expo: muro
  await page.goto(`${BASE}/expo/pantalla`);
  await page.waitForSelector('[data-testid="wall-empty"]');
  await page.waitForTimeout(500);
  await shot(page, viewport, '16-muro-vacio');
  const qrBuffer = await page.locator('[data-testid="wall-qr"]').screenshot();
  const qrPng = PNG.sync.read(qrBuffer);
  const decoded = jsQR(new Uint8ClampedArray(qrPng.data), qrPng.width, qrPng.height);
  if (decoded?.data !== 'http://192.168.0.10:8080/expo/subir') issues.push(`[${name}] QR no legible: ${decoded?.data}`);
  else notes.push(`[${name}] QR decodificado: ${decoded.data} (${qrPng.width}px)`);
  for (const count of [1, 3, 5, 9]) {
    await set({ gallery: count });
    await page.waitForFunction((n) => document.querySelectorAll('[data-testid="wall-tile"]').length === n, count, { timeout: 8000 });
    await page.waitForTimeout(1500);
    await shot(page, viewport, `17-muro-${count}`, { check: count === 9 });
  }
  const muted = await page.locator('[data-testid="wall-tile"] video').first().evaluate((v) => ({ muted: v.muted, loop: v.loop, paused: v.paused }));
  if (!muted.muted || !muted.loop || muted.paused) issues.push(`[${name}] video del muro: ${JSON.stringify(muted)}`);
  await set({ apiDown: true });
  await page.waitForSelector('[data-testid="wall-offline"]', { timeout: 8000 });
  await shot(page, viewport, '18-muro-sin-conexion', { check: false });
  if ((await page.locator('[data-testid="wall-tile"]').count()) !== 9) issues.push(`[${name}] el muro se vació sin conexión`);
  await set(defaults);

  // Sin configuración
  await set({ apiDown: true });
  await page.goto(`${BASE}/`);
  await page.waitForSelector('text=Reintentando');
  await shot(page, viewport, '19-sin-servidor');
  await set(defaults);
  await page.waitForSelector('[data-testid="dropzone"]', { timeout: 8000 });

  await context.close();
}

// Movimiento reducido: nada de GSAP en marcha
{
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, reducedMotion: 'reduce' });
  const page = await context.newPage();
  await set(defaults);
  await page.goto(`${BASE}/`);
  await page.waitForSelector('[data-testid="dropzone"]');
  await page.waitForTimeout(1500);
  const moved = await page.evaluate(() =>
    [...document.querySelectorAll('.letter, .orb')].filter((el) => getComputedStyle(el).transform !== 'none').length,
  );
  if (moved) issues.push(`movimiento reducido: ${moved} elementos con transform`);
  else notes.push('Movimiento reducido: portada sin animaciones.');
  await context.close();
}

await browser.close();
mock.kill();

console.log(notes.join('\n'));
if (issues.length) {
  console.log(`\nPROBLEMAS (${issues.length}):\n- ${[...new Set(issues)].join('\n- ')}`);
  process.exitCode = 1;
} else {
  console.log('\nVerificación visual sin problemas.');
}
console.log(`Capturas en ${SHOTS}`);
