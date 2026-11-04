// Datos de prueba compartidos. Solo existen en archivos *.spec.ts.
import { HttpTestingController } from '@angular/common/http/testing';
import { AnalyzeResponse, AppConfig, JobResponse } from '../core/api.models';
import { ConfigStore } from '../core/stores';
import { TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';

// La app no usa Zone.js; las pruebas corren igual que producción.
beforeEach(() => {
  TestBed.configureTestingModule({ providers: [provideZonelessChangeDetection()] });
});

export const TEST_CONFIG: AppConfig = {
  public_base_url: 'http://10.1.2.3:8080/',
  limits: {
    max_upload_bytes: 1000,
    max_video_seconds: 7,
    min_side_px: 300,
    formats: ['jpg', 'png', 'webp', 'mp4', 'mov'],
  },
  moods: ['calma', 'epico', 'mood_nuevo'],
  classes: [
    { id: 0, name: 'fondo', label: 'Etiqueta 0', color: '#101010' },
    { id: 1, name: 'cielo', label: 'Etiqueta 1', color: '#112233' },
    { id: 3, name: 'vegetacion', label: 'Etiqueta 3', color: '#445566' },
    { id: 4, name: 'fuego_luz', label: 'Etiqueta 4', color: '#778899' },
    { id: 9, name: 'clase_futura', label: 'Etiqueta 9', color: '#ABCDEF' },
  ],
};

export function testAnalysis(overrides: Partial<AnalyzeResponse> = {}): AnalyzeResponse {
  return {
    width: 640,
    height: 480,
    mask_png_base64: '',
    classes: [
      { id: 9, name: 'clase_futura', pixel_pct: 73.37, confidence: 0.812 },
      { id: 4, name: 'fuego_luz', pixel_pct: 11.5, confidence: 0.377 },
    ],
    inference_ms: { segmentation: 321.44, depth: null, total: 987.65 },
    depth_png_base64: null,
    model_version: 'prueba-7',
    model_sha256: 'f'.repeat(64),
    conf_threshold: 0.42,
    ...overrides,
  };
}

export function testJob(overrides: Partial<JobResponse> = {}): JobResponse {
  return {
    job_id: 'job-1',
    status: 'running',
    stage: 'rendering',
    progress: 0.25,
    mood: 'calma',
    seed: 7,
    created_at: '2026-01-01T00:00:00Z',
    updated_at: '2026-01-01T00:00:00Z',
    result: null,
    timings_ms: null,
    error: null,
    ...overrides,
  };
}

/** Carga /config en el store para que los componentes la lean. */
export function provideLoadedConfig(http: HttpTestingController, config = TEST_CONFIG): void {
  TestBed.inject(ConfigStore).load();
  http.expectOne('/api/v1/config').flush(config);
}

function crc32(bytes: Uint8Array): number {
  let crc = ~0;
  for (const byte of bytes) {
    crc ^= byte;
    for (let k = 0; k < 8; k++) crc = crc & 1 ? (crc >>> 1) ^ 0xedb88320 : crc >>> 1;
  }
  return ~crc >>> 0;
}

function chunk(type: string, data: Uint8Array): Uint8Array {
  const out = new Uint8Array(12 + data.length);
  const view = new DataView(out.buffer);
  view.setUint32(0, data.length);
  for (let i = 0; i < 4; i++) out[4 + i] = type.charCodeAt(i);
  out.set(data, 8);
  view.setUint32(8 + data.length, crc32(out.subarray(4, 8 + data.length)));
  return out;
}

/** PNG real de 8 bits y 1 canal, la forma exacta de `mask_png_base64`. */
export async function grayscalePng(width: number, height: number, values: number[]): Promise<Uint8Array> {
  const raw = new Uint8Array((width + 1) * height);
  for (let y = 0; y < height; y++) {
    raw[y * (width + 1)] = 0;
    raw.set(values.slice(y * width, (y + 1) * width), y * (width + 1) + 1);
  }
  const compressed = await new Response(
    new Blob([raw]).stream().pipeThrough(new CompressionStream('deflate')),
  ).arrayBuffer();
  const ihdr = new Uint8Array(13);
  const view = new DataView(ihdr.buffer);
  view.setUint32(0, width);
  view.setUint32(4, height);
  ihdr[8] = 8;
  ihdr[9] = 0;
  const parts = [
    Uint8Array.of(0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a),
    chunk('IHDR', ihdr),
    chunk('IDAT', new Uint8Array(compressed)),
    chunk('IEND', new Uint8Array(0)),
  ];
  const png = new Uint8Array(parts.reduce((sum, part) => sum + part.length, 0));
  let offset = 0;
  for (const part of parts) {
    png.set(part, offset);
    offset += part.length;
  }
  return png;
}

export function toBase64(bytes: Uint8Array): string {
  return btoa(String.fromCharCode(...bytes));
}

const MAGIC_BYTES: Record<string, number[]> = {
  jpg: [0xff, 0xd8, 0xff, 0xe0, 0, 0, 0, 0, 0, 0, 0, 0],
  png: [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0, 0, 0, 0],
  webp: [0x52, 0x49, 0x46, 0x46, 0, 0, 0, 0, 0x57, 0x45, 0x42, 0x50],
  mp4: [0, 0, 0, 0x18, 0x66, 0x74, 0x79, 0x70, 0x69, 0x73, 0x6f, 0x6d],
  mov: [0, 0, 0, 0x14, 0x66, 0x74, 0x79, 0x70, 0x71, 0x74, 0x20, 0x20],
  heic: [0, 0, 0, 0x18, 0x66, 0x74, 0x79, 0x70, 0x68, 0x65, 0x69, 0x63],
};

export function fileWithMagic(format: string, name = `archivo.${format}`, size = 64): File {
  const body = new Uint8Array(size);
  body.set((MAGIC_BYTES[format] ?? []).slice(0, size));
  return new File([body], name);
}
