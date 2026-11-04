import { AppLimits } from './api.models';
import { ApiError } from './api-error';

export type MediaKind = 'image' | 'video';

export interface ValidatedMedia {
  file: File;
  kind: MediaKind;
  format: string;
  /** Sin dimensiones: el navegador no decodifica ese video (p. ej. .mov en Chrome). */
  width: number | null;
  height: number | null;
  durationSeconds: number | null;
}

export type ValidationResult =
  | { ok: true; media: ValidatedMedia }
  | { ok: false; error: ApiError };

export interface MediaProbe {
  image(file: File): Promise<{ width: number; height: number } | null>;
  video(file: File): Promise<{ width: number; height: number; duration: number } | null>;
}

const VIDEO_FORMATS = new Set(['mp4', 'mov']);
const QUICKTIME_ATOMS = new Set(['moov', 'mdat', 'wide', 'free', 'skip', 'pnot']);
const HEIF_BRANDS = new Set(['heic', 'heix', 'hevc', 'hevx', 'heim', 'heis', 'mif1', 'msf1', 'avif']);

function ascii(bytes: Uint8Array, start: number, end: number): string {
  return String.fromCharCode(...bytes.subarray(start, end));
}

/** Reconoce el formato por magic bytes, igual que el servidor; no confía en la extensión. */
export function sniffFormat(bytes: Uint8Array): string | null {
  if (bytes.length >= 3 && bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff) return 'jpg';
  if (bytes.length >= 8 && ascii(bytes, 1, 4) === 'PNG' && bytes[0] === 0x89) return 'png';
  if (bytes.length >= 12 && ascii(bytes, 0, 4) === 'RIFF' && ascii(bytes, 8, 12) === 'WEBP') {
    return 'webp';
  }
  if (bytes.length >= 12) {
    const atom = ascii(bytes, 4, 8);
    if (atom === 'ftyp') {
      const brand = ascii(bytes, 8, 12);
      if (brand === 'qt  ') return 'mov';
      return HEIF_BRANDS.has(brand) ? null : 'mp4';
    }
    if (QUICKTIME_ATOMS.has(atom)) return 'mov';
  }
  return null;
}

export function formatBytes(bytes: number): string {
  const mb = bytes / (1024 * 1024);
  return `${mb >= 10 ? Math.round(mb) : Math.round(mb * 10) / 10} MB`;
}

export function formatList(formats: readonly string[]): string {
  return formats.join(', ');
}

/** Atributo `accept` del selector de archivos, derivado de los formatos de /config. */
export function acceptFromFormats(formats: readonly string[]): string {
  return formats
    .flatMap((format) => (format === 'jpg' ? ['.jpg', '.jpeg'] : [`.${format}`]))
    .join(',');
}

/** Sondeo real con el decodificador del navegador. */
export const browserProbe: MediaProbe = {
  image(file) {
    return new Promise((resolve) => {
      const url = URL.createObjectURL(file);
      const img = new Image();
      img.onload = () => {
        URL.revokeObjectURL(url);
        resolve({ width: img.naturalWidth, height: img.naturalHeight });
      };
      img.onerror = () => {
        URL.revokeObjectURL(url);
        resolve(null);
      };
      img.src = url;
    });
  },
  video(file) {
    return new Promise((resolve) => {
      const url = URL.createObjectURL(file);
      const video = document.createElement('video');
      const finish = (value: { width: number; height: number; duration: number } | null) => {
        clearTimeout(guard);
        URL.revokeObjectURL(url);
        video.removeAttribute('src');
        resolve(value);
      };
      const guard = setTimeout(() => finish(null), 8000);
      video.preload = 'metadata';
      video.muted = true;
      video.onloadedmetadata = () =>
        finish({ width: video.videoWidth, height: video.videoHeight, duration: video.duration });
      video.onerror = () => finish(null);
      video.src = url;
    });
  },
};

/**
 * Validación previa, en el orden del contrato: tamaño, formato, decodificación,
 * resolución, duración. Ahorra una subida fallida; el servidor sigue siendo quien decide.
 */
export async function validateUpload(
  file: File,
  limits: AppLimits,
  probe: MediaProbe = browserProbe,
): Promise<ValidationResult> {
  const reject = (code: string, message: string): ValidationResult => ({
    ok: false,
    error: new ApiError(code, message),
  });

  if (file.size === 0) return reject('CORRUPT_FILE', 'El archivo está vacío.');
  if (file.size > limits.max_upload_bytes) {
    return reject(
      'FILE_TOO_LARGE',
      `El archivo pesa ${formatBytes(file.size)}; el máximo es ${formatBytes(limits.max_upload_bytes)}.`,
    );
  }

  const head = new Uint8Array(await file.slice(0, 16).arrayBuffer());
  const format = sniffFormat(head);
  if (!format || !limits.formats.includes(format)) {
    return reject(
      'UNSUPPORTED_FORMAT',
      `Este archivo no es compatible. Formatos admitidos: ${formatList(limits.formats)}.`,
    );
  }

  const kind: MediaKind = VIDEO_FORMATS.has(format) ? 'video' : 'image';
  const lowResolution = (width: number, height: number) =>
    reject(
      'RESOLUTION_TOO_LOW',
      `Mide ${width}×${height} px; el lado menor debe tener al menos ${limits.min_side_px} px.`,
    );

  if (kind === 'image') {
    const size = await probe.image(file);
    if (!size) return reject('CORRUPT_FILE', 'No pudimos abrir esta imagen; puede estar dañada.');
    if (Math.min(size.width, size.height) < limits.min_side_px) {
      return lowResolution(size.width, size.height);
    }
    return {
      ok: true,
      media: { file, kind, format, width: size.width, height: size.height, durationSeconds: null },
    };
  }

  // Si el navegador no decodifica el video, no se puede juzgar aquí: lo valida el servidor.
  const meta = await probe.video(file);
  if (meta) {
    if (meta.width > 0 && Math.min(meta.width, meta.height) < limits.min_side_px) {
      return lowResolution(meta.width, meta.height);
    }
    if (Number.isFinite(meta.duration) && meta.duration > limits.max_video_seconds) {
      return reject(
        'VIDEO_TOO_LONG',
        `El video dura ${Math.ceil(meta.duration)} s; el máximo es ${limits.max_video_seconds} s.`,
      );
    }
  }
  return {
    ok: true,
    media: {
      file,
      kind,
      format,
      width: meta?.width || null,
      height: meta?.height || null,
      durationSeconds: meta?.duration ?? null,
    },
  };
}
