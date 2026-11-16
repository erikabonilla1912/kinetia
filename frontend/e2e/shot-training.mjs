// SOLO PRUEBAS. Captura /entrenamiento y /modelo con la compilación servida localmente y la
// API real del clúster (http://localhost:8080). Comprueba scroll horizontal y contraste (axe).
// Requiere `npm run build` y el clúster en marcha. Uso: node e2e/shot-training.mjs
import { createServer, request } from 'node:http';
import { existsSync, mkdirSync, readFileSync, statSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, extname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const here = dirname(fileURLToPath(import.meta.url));
const DIST = resolve(here, '../dist/frontend/browser');
const SHOTS = resolve(here, '../.screenshots');
mkdirSync(SHOTS, { recursive: true });
const axe = readFileSync(createRequire(import.meta.url).resolve('axe-core/axe.min.js'), 'utf8');
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.svg': 'image/svg+xml', '.woff2': 'font/woff2' };

const server = createServer((req, res) => {
  if (req.url.startsWith('/api/')) {
    const up = request({ host: 'localhost', port: 8080, path: req.url, method: req.method, headers: req.headers }, (r) => {
      res.writeHead(r.statusCode, r.headers);
      r.pipe(res);
    });
    req.pipe(up);
    return;
  }
  let file = join(DIST, decodeURIComponent(req.url.split('?')[0]));
  if (!existsSync(file) || statSync(file).isDirectory()) file = join(DIST, 'index.html');
  res.writeHead(200, { 'content-type': TYPES[extname(file)] ?? 'application/octet-stream' });
  res.end(readFileSync(file));
});
await new Promise((done) => server.listen(0, '127.0.0.1', done));
const BASE = `http://127.0.0.1:${server.address().port}`;

const browser = await chromium.launch();
const problems = [];
for (const vp of [
  { name: '390', width: 390, height: 844, isMobile: true, hasTouch: true, deviceScaleFactor: 2 },
  { name: '1440', width: 1440, height: 900 },
]) {
  const page = await browser.newPage({ viewport: { width: vp.width, height: vp.height }, ...vp });
  for (const route of ['/entrenamiento', '/modelo']) {
    await page.goto(BASE + route, { waitUntil: 'networkidle' });
    await page.waitForTimeout(800);
    const slug = route.slice(1);
    await page.screenshot({ path: join(SHOTS, `train-${slug}-${vp.name}.png`), fullPage: true });
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
    if (overflow > 0) problems.push(`${route} @${vp.name}: scroll horizontal de ${overflow}px`);
    await page.addScriptTag({ content: axe });
    const result = await page.evaluate(() => window.axe.run(document, { runOnly: ['wcag2a', 'wcag2aa'] }));
    for (const v of result.violations) problems.push(`${route} @${vp.name}: ${v.id} (${v.nodes.length})`);
  }
  await page.close();
}
await browser.close();
server.close();
console.log(problems.length ? problems.join('\n') : 'SIN PROBLEMAS');
console.log('Capturas en', SHOTS);
