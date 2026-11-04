import { TEST_CONFIG, fileWithMagic } from '../testing/fixtures.spec';
import { MediaProbe, acceptFromFormats, sniffFormat, validateUpload } from './upload-validator';

const limits = TEST_CONFIG.limits;

const probe = (
  image: { width: number; height: number } | null = { width: 800, height: 600 },
  video: { width: number; height: number; duration: number } | null = { width: 800, height: 600, duration: 3 },
): MediaProbe => ({
  image: () => Promise.resolve(image),
  video: () => Promise.resolve(video),
});

async function errorCode(file: File, media = probe()): Promise<string | null> {
  const result = await validateUpload(file, limits, media);
  return result.ok ? null : result.error.code;
}

describe('validateUpload', () => {
  it('acepta una imagen válida y conserva sus medidas', async () => {
    const result = await validateUpload(fileWithMagic('png'), limits, probe());
    expect(result.ok).toBeTrue();
    if (result.ok) {
      expect(result.media.kind).toBe('image');
      expect(result.media.format).toBe('png');
      expect(result.media.width).toBe(800);
    }
  });

  it('rechaza un archivo vacío', async () => {
    expect(await errorCode(new File([], 'vacio.png'))).toBe('CORRUPT_FILE');
  });

  it('rechaza por tamaño usando el límite de /config y lo menciona', async () => {
    const result = await validateUpload(fileWithMagic('png', 'grande.png', 1001), limits, probe());
    expect(result.ok).toBeFalse();
    if (!result.ok) expect(result.error.code).toBe('FILE_TOO_LARGE');
    expect(await errorCode(fileWithMagic('png', 'justo.png', 1000))).toBeNull();
  });

  it('decide por magic bytes y no por la extensión', async () => {
    const fake = new File(['no soy una imagen, pero termino en jpg'], 'foto.jpg');
    expect(await errorCode(fake)).toBe('UNSUPPORTED_FORMAT');
    expect(await errorCode(fileWithMagic('png', 'foto.txt'))).toBeNull();
  });

  it('rechaza HEIC como formato no compatible', async () => {
    expect(await errorCode(fileWithMagic('heic'))).toBe('UNSUPPORTED_FORMAT');
  });

  it('solo admite los formatos que dice /config', async () => {
    const result = await validateUpload(
      fileWithMagic('webp'),
      { ...limits, formats: ['png'] },
      probe(),
    );
    expect(result.ok).toBeFalse();
    if (!result.ok) expect(result.error.message).toContain('png');
  });

  it('marca como corrupta una imagen que el navegador no decodifica', async () => {
    expect(await errorCode(fileWithMagic('jpg'), probe(null))).toBe('CORRUPT_FILE');
  });

  it('rechaza una resolución menor que min_side_px de /config', async () => {
    const result = await validateUpload(fileWithMagic('jpg'), limits, probe({ width: 299, height: 900 }));
    expect(result.ok).toBeFalse();
    if (!result.ok) {
      expect(result.error.code).toBe('RESOLUTION_TOO_LOW');
      expect(result.error.message).toContain('300');
    }
  });

  it('rechaza un video más largo que max_video_seconds de /config', async () => {
    const result = await validateUpload(
      fileWithMagic('mp4'),
      limits,
      probe(undefined, { width: 800, height: 600, duration: 7.5 }),
    );
    expect(result.ok).toBeFalse();
    if (!result.ok) {
      expect(result.error.code).toBe('VIDEO_TOO_LONG');
      expect(result.error.message).toContain('7 s');
    }
  });

  it('deja pasar al servidor un video que este navegador no puede leer', async () => {
    const result = await validateUpload(fileWithMagic('mov'), limits, probe(undefined, null));
    expect(result.ok).toBeTrue();
    if (result.ok) {
      expect(result.media.kind).toBe('video');
      expect(result.media.width).toBeNull();
    }
  });

  it('respeta el orden del contrato: tamaño antes que formato', async () => {
    const huge = new File([new Uint8Array(2000)], 'x.bin');
    expect(await errorCode(huge)).toBe('FILE_TOO_LARGE');
  });
});

describe('sniffFormat', () => {
  it('reconoce cada formato del contrato', async () => {
    for (const format of ['jpg', 'png', 'webp', 'mp4', 'mov']) {
      const bytes = new Uint8Array(await fileWithMagic(format).arrayBuffer());
      expect(sniffFormat(bytes)).withContext(format).toBe(format);
    }
  });

  it('devuelve null con datos cortos o desconocidos', () => {
    expect(sniffFormat(new Uint8Array(0))).toBeNull();
    expect(sniffFormat(new Uint8Array(32))).toBeNull();
  });
});

describe('acceptFromFormats', () => {
  it('deriva las extensiones de /config', () => {
    expect(acceptFromFormats(['jpg', 'mov'])).toBe('.jpg,.jpeg,.mov');
  });
});
