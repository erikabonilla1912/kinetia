// SOLO PRUEBAS. Crea e2e/.fixtures: fotos sintéticas, archivos inválidos y MP4 H.264 de prueba.
// Los MP4 los graba Chrome con MediaRecorder, para no depender de FFmpeg.
import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { renderScene } from './fixtures.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const OUT = resolve(here, '.fixtures');
mkdirSync(OUT, { recursive: true });

const scenes = ['lake', 'forest', 'night'];
for (const scene of scenes) {
  const path = join(OUT, `${scene}.png`);
  if (!existsSync(path)) writeFileSync(path, renderScene(1280, 720, scene));
}
if (!existsSync(join(OUT, 'portrait.png'))) writeFileSync(join(OUT, 'portrait.png'), renderScene(720, 1280, 'lake'));
writeFileSync(join(OUT, 'tiny.png'), renderScene(120, 90, 'lake'));
writeFileSync(join(OUT, 'not-an-image.jpg'), Buffer.from('esto no es una imagen, solo texto con extensión jpg'));

const hasVideo = (scene) => existsSync(join(OUT, `result-${scene}.mp4`)) && statSync(join(OUT, `result-${scene}.mp4`)).size > 0;
// La primera grabación de Chrome a veces sale vacía mientras arranca el codificador.
const missing = [...scenes.filter((scene) => !hasVideo(scene)), ...scenes.filter((scene) => !hasVideo(scene))];
if (missing.length) {
  const browser = await chromium.launch({ channel: 'chrome' });
  for (const scene of missing) {
    if (hasVideo(scene)) continue;
    const page = await browser.newPage();
    const dataUrl = `data:image/png;base64,${readFileSync(join(OUT, `${scene}.png`)).toString('base64')}`;
    const base64 = await page.evaluate(async (src) => {
      const img = new Image();
      img.src = src;
      await img.decode();
      const canvas = document.createElement('canvas');
      canvas.width = 1280;
      canvas.height = 720;
      const ctx = canvas.getContext('2d');
      const stream = canvas.captureStream(24);
      const recorder = new MediaRecorder(stream, { mimeType: 'video/mp4;codecs=avc1.42E01F', videoBitsPerSecond: 2_500_000 });
      const chunks = [];
      recorder.ondataavailable = (e) => chunks.push(e.data);
      const stopped = new Promise((done) => (recorder.onstop = done));
      recorder.start();
      const started = performance.now();
      await new Promise((done) => {
        const draw = () => {
          const t = (performance.now() - started) / 1000;
          const shift = Math.sin((t / 3) * Math.PI * 2) * 22;
          ctx.drawImage(img, 0, 0, 1280, 420, shift - 30, 0, 1340, 420);
          for (let y = 420; y < 720; y += 2) {
            const dx = Math.sin(y * 0.07 + t * 5) * ((y - 420) / 300) * 9;
            ctx.drawImage(img, 0, y, 1280, 2, dx, y, 1280, 2);
          }
          if (t < 3) requestAnimationFrame(draw);
          else done();
        };
        draw();
      });
      recorder.stop();
      await stopped;
      const buffer = await new Blob(chunks).arrayBuffer();
      let binary = '';
      new Uint8Array(buffer).forEach((b) => (binary += String.fromCharCode(b)));
      return btoa(binary);
    }, dataUrl);
    writeFileSync(join(OUT, `result-${scene}.mp4`), Buffer.from(base64, 'base64'));
    await page.close();
  }
  await browser.close();
}
console.log('Fixtures listos en', OUT);
