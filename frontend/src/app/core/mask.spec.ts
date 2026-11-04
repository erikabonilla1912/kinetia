import { TEST_CONFIG, grayscalePng, toBase64 } from '../testing/fixtures.spec';
import { decodeMask, hexToRgb, paintMask } from './mask';

describe('decodeMask', () => {
  it('recupera los índices exactos de un PNG de 8 bits y 1 canal', async () => {
    const values = [0, 1, 2, 3, 4, 5, 6, 0];
    const png = await grayscalePng(4, 2, values);
    const mask = await decodeMask(toBase64(png));
    expect(mask.width).toBe(4);
    expect(mask.height).toBe(2);
    expect([...mask.indices]).toEqual(values);
  });
});

describe('paintMask', () => {
  const mask = { width: 3, height: 1, indices: Uint8Array.of(1, 3, 200) };

  it('usa los colores de /config y deja transparente lo desconocido', () => {
    const image = paintMask(mask, TEST_CONFIG.classes);
    expect([...image.data.slice(0, 4)]).toEqual([0x11, 0x22, 0x33, 255]);
    expect([...image.data.slice(4, 8)]).toEqual([0x44, 0x55, 0x66, 255]);
    expect(image.data[11]).toBe(0);
  });

  it('con focusId solo deja visible esa clase', () => {
    const image = paintMask(mask, TEST_CONFIG.classes, 3);
    expect(image.data[3]).toBe(0);
    expect(image.data[7]).toBe(255);
  });
});

describe('hexToRgb', () => {
  it('convierte y rechaza valores inválidos', () => {
    expect(hexToRgb('#FF7043')).toEqual([255, 112, 67]);
    expect(hexToRgb('rojo')).toBeNull();
  });
});
