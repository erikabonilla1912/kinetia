// SOLO PRUEBAS. Genera imágenes sintéticas y sus máscaras para la verificación visual.
// Nada de este directorio entra en el build de producción.
import { PNG } from 'pngjs';

const CLASS = { fondo: 0, cielo: 1, agua: 2, vegetacion: 3, fuego_luz: 4, persona: 5, animal: 6 };

function ridge(u, a, b, c) {
  return a + 0.05 * Math.sin(u * 7 + b) + 0.025 * Math.sin(u * 19 + c) + 0.012 * Math.sin(u * 47 + b);
}

/** Clase de cada punto normalizado de la escena. Es la misma para la foto y la máscara. */
export function classify(u, v, aspect, scene = 'lake') {
  const horizon = scene === 'forest' ? 0.68 : 0.58;
  const dx = (u - 0.68) * aspect;
  // Figura de pie.
  if (Math.hypot(dx, v - 0.5) < 0.028) return CLASS.persona;
  if (Math.abs(dx) < 0.03 && v > 0.52 && v < 0.74) return CLASS.persona;
  // Animal junto a la figura.
  const ax = (u - 0.76) * aspect;
  if (Math.abs(ax) < 0.036 && v > 0.69 && v < 0.74) return CLASS.animal;
  // Fogata.
  const fx = (u - 0.56) * aspect;
  if (Math.abs(fx) < 0.05 * (1 - (0.76 - v) / 0.13) && v > 0.63 && v < 0.76) return CLASS.fuego_luz;
  // Orilla con pasto.
  const shore = 0.74 + 0.02 * Math.sin(u * 13);
  if (u > 0.4 && v > shore) return CLASS.vegetacion;
  // Árboles.
  const treeLine = 0.34 + 0.28 * u + 0.05 * Math.sin(u * 42) + 0.02 * Math.sin(u * 111);
  if (u < 0.42 && v > treeLine) return CLASS.vegetacion;
  if (v > horizon) return CLASS.agua;
  if (v > ridge(u, 0.42, 1.2, 0.3)) return CLASS.fondo;
  return CLASS.cielo;
}

const PALETTES = {
  lake: { top: [28, 34, 92], mid: [214, 92, 96], low: [255, 178, 104], water: [36, 60, 110], mount: [42, 36, 74] },
  forest: { top: [84, 148, 214], mid: [170, 214, 232], low: [236, 228, 196], water: [40, 104, 118], mount: [52, 92, 88] },
  night: { top: [6, 8, 28], mid: [18, 26, 66], low: [52, 44, 104], water: [10, 18, 44], mount: [10, 12, 30] },
};

function hash(x, y) {
  const s = Math.sin(x * 12.9898 + y * 78.233) * 43758.5453;
  return s - Math.floor(s);
}

const lerp = (a, b, t) => a + (b - a) * t;
const mix3 = (a, b, t) => [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)];

function shade(u, v, cls, aspect, px, py, scene) {
  const p = PALETTES[scene] ?? PALETTES.lake;
  const sky = (t) => (t < 0.55 ? mix3(p.top, p.mid, t / 0.55) : mix3(p.mid, p.low, (t - 0.55) / 0.45));
  switch (cls) {
    case CLASS.cielo: {
      let c = sky(Math.min(1, v / 0.58));
      const sunD = Math.hypot((u - 0.68) * aspect, v - 0.4);
      if (scene !== 'night') c = mix3(c, [255, 232, 170], Math.max(0, 1 - sunD / 0.3) ** 2.2);
      else if (hash(px, py) > 0.9975) c = [255, 255, 245];
      const cloud = Math.sin(u * 9 + v * 30) * Math.sin(u * 3.4 - v * 12);
      return mix3(c, [255, 214, 196], Math.max(0, cloud - 0.45) * 0.45);
    }
    case CLASS.agua: {
      const t = Math.min(1, (v - 0.58) / 0.42);
      let c = mix3(sky(1 - t * 0.9), p.water, 0.35 + t * 0.55);
      const wave = Math.sin(v * 260 + Math.sin(u * 22) * 3);
      c = c.map((x) => x + wave * 9);
      const glint = Math.max(0, 1 - Math.abs(u - 0.68) * aspect * 6) * (0.5 + wave * 0.5);
      return scene === 'night' ? c : mix3(c, [255, 216, 150], glint * 0.6);
    }
    case CLASS.fondo:
      return mix3(p.mount, sky(0.7), 0.12 + (0.58 - v) * 0.5);
    case CLASS.vegetacion: {
      const n = hash(Math.floor(px / 3), Math.floor(py / 3));
      return mix3(scene === 'night' ? [10, 26, 22] : [20, 62, 44], scene === 'night' ? [18, 48, 36] : [62, 132, 74], n * 0.75);
    }
    case CLASS.fuego_luz:
      return mix3([255, 76, 30], [255, 224, 120], Math.min(1, Math.max(0, (v - 0.63) / 0.13)));
    case CLASS.persona:
      return [18, 16, 28];
    case CLASS.animal:
      return [38, 26, 22];
    default:
      return [0, 0, 0];
  }
}

export function renderScene(width, height, scene = 'lake') {
  const png = new PNG({ width, height, colorType: 2 });
  const aspect = width / height;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const u = x / width;
      const v = y / height;
      const [r, g, b] = shade(u, v, classify(u, v, aspect, scene), aspect, x, y, scene);
      const grain = (hash(x, y) - 0.5) * 6;
      const i = (y * width + x) * 4;
      png.data[i] = Math.max(0, Math.min(255, r + grain));
      png.data[i + 1] = Math.max(0, Math.min(255, g + grain));
      png.data[i + 2] = Math.max(0, Math.min(255, b + grain));
      png.data[i + 3] = 255;
    }
  }
  return PNG.sync.write(png, { colorType: 2 });
}

/** PNG de 8 bits, 1 canal, valor = índice de clase: la forma exacta del contrato. */
export function renderMask(width, height, scene = 'lake') {
  const png = new PNG({ width, height, colorType: 0, inputColorType: 0, bitDepth: 8, inputHasAlpha: false });
  const data = Buffer.alloc(width * height);
  const counts = new Map();
  const aspect = width / height;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const cls = classify(x / width, y / height, aspect, scene);
      data[y * width + x] = cls;
      counts.set(cls, (counts.get(cls) ?? 0) + 1);
    }
  }
  png.data = data;
  const buffer = PNG.sync.write(png, { colorType: 0, inputColorType: 0, bitDepth: 8, inputHasAlpha: false });
  return { buffer, counts };
}
