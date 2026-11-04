import { ApiError, NETWORK_ERROR, UNKNOWN_ERROR } from './api-error';

export type Recovery = 'retry' | 'change-file' | 'none';

export interface ErrorGuide {
  title: string;
  hint: string;
  recovery: Recovery;
}

/** Una guía por cada `code` del contrato (sección 4), más los del navegador. */
export const ERROR_GUIDES: Readonly<Record<string, ErrorGuide>> = {
  MISSING_FILE: {
    title: 'Falta el archivo',
    hint: 'Elige una foto o un video antes de continuar.',
    recovery: 'change-file',
  },
  CORRUPT_FILE: {
    title: 'No pudimos abrir el archivo',
    hint: 'Puede estar dañado. Prueba con otra foto o vuelve a exportarla.',
    recovery: 'change-file',
  },
  UNAUTHORIZED: {
    title: 'Acceso restringido',
    hint: 'Esta acción es solo para el equipo de moderación.',
    recovery: 'none',
  },
  JOB_NOT_FOUND: {
    title: 'Esta animación ya no existe',
    hint: 'Vuelve a crearla con la misma foto.',
    recovery: 'retry',
  },
  RESULT_NOT_FOUND: {
    title: 'El resultado ya no está disponible',
    hint: 'Pudo haber expirado. Vuelve a crear la animación.',
    recovery: 'retry',
  },
  METRICS_NOT_FOUND: {
    title: 'Aún no hay métricas publicadas',
    hint: 'Aparecerán aquí cuando el modelo evaluado incluya su metrics.json.',
    recovery: 'retry',
  },
  JOB_NOT_READY: {
    title: 'La animación todavía no termina',
    hint: 'Espera unos segundos y vuelve a intentarlo.',
    recovery: 'retry',
  },
  FILE_TOO_LARGE: {
    title: 'El archivo pesa demasiado',
    hint: 'Elige uno más liviano o reduce su calidad.',
    recovery: 'change-file',
  },
  UNSUPPORTED_FORMAT: {
    title: 'Formato no compatible',
    hint: 'Convierte el archivo a uno de los formatos admitidos y súbelo otra vez.',
    recovery: 'change-file',
  },
  RESOLUTION_TOO_LOW: {
    title: 'La imagen es muy pequeña',
    hint: 'Usa una foto con más resolución para que el modelo vea los detalles.',
    recovery: 'change-file',
  },
  RESOLUTION_TOO_HIGH: {
    title: 'La imagen es demasiado grande',
    hint: 'Redúcela a menos megapíxeles y vuelve a subirla.',
    recovery: 'change-file',
  },
  VIDEO_TOO_LONG: {
    title: 'El video es muy largo',
    hint: 'Recórtalo o graba uno más corto.',
    recovery: 'change-file',
  },
  INVALID_PARAMS: {
    title: 'Algún ajuste no es válido',
    hint: 'Revisa el mood, la duración y el pincel. Si dibujaste mucho, borra algún trazo.',
    recovery: 'retry',
  },
  QUEUE_FULL: {
    title: 'La cola está llena',
    hint: 'Hay muchas animaciones en proceso. Tu foto sigue aquí: reintenta en un momento.',
    recovery: 'retry',
  },
  MODEL_NOT_READY: {
    title: 'El modelo está cargando',
    hint: 'Pasa justo después de encender el sistema. Espera un momento y vuelve a intentarlo.',
    recovery: 'retry',
  },
  INTERNAL_ERROR: {
    title: 'Algo falló de nuestro lado',
    hint: 'No fue tu archivo. Vuelve a intentarlo; si persiste, avísale al equipo.',
    recovery: 'retry',
  },
  NOT_FOUND: {
    title: 'Esa dirección no existe',
    hint: 'Vuelve al estudio y empieza de nuevo. Si persiste, avísale al equipo.',
    recovery: 'none',
  },
  METHOD_NOT_ALLOWED: {
    title: 'El servidor rechazó la operación',
    hint: 'Es un fallo de la aplicación, no de tu archivo. Avísale al equipo.',
    recovery: 'none',
  },
  RANGE_NOT_SATISFIABLE: {
    title: 'No se pudo reproducir el resultado',
    hint: 'Recarga la página para volver a pedir el archivo.',
    recovery: 'retry',
  },
  ENGINE_UNAVAILABLE: {
    title: 'El motor de animación no está disponible',
    hint: 'No fue tu archivo. Avísale al equipo; Rayos X sigue funcionando.',
    recovery: 'retry',
  },
  ENCODER_UNAVAILABLE: {
    title: 'No se pudo codificar el video',
    hint: 'No fue tu archivo: falta una herramienta en el servidor. Avísale al equipo.',
    recovery: 'retry',
  },
  JOB_INTERRUPTED: {
    title: 'La animación se interrumpió',
    hint: 'El servidor se reinició mientras trabajaba. Vuelve a lanzarla.',
    recovery: 'retry',
  },
  [NETWORK_ERROR]: {
    title: 'Sin conexión con el servidor',
    hint: 'Revisa que sigas conectado a la red de KINETIA y vuelve a intentarlo.',
    recovery: 'retry',
  },
  [UNKNOWN_ERROR]: {
    title: 'Ocurrió un error inesperado',
    hint: 'Vuelve a intentarlo en unos segundos.',
    recovery: 'retry',
  },
};

export interface ErrorView extends ErrorGuide {
  code: string;
  message: string;
  retryAfterSeconds: number | null;
}

/** Une la guía por `code` con el `message` del servidor, que se muestra literal. */
export function describeError(error: ApiError): ErrorView {
  const guide = ERROR_GUIDES[error.code] ?? ERROR_GUIDES[UNKNOWN_ERROR];
  return {
    ...guide,
    code: error.code,
    message: error.message,
    retryAfterSeconds: error.retryAfterSeconds,
  };
}
