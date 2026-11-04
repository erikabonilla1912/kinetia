/**
 * Solo textos y estilo visual de los moods. Cuáles existen lo dicta /config. Las clases
 * no están aquí: sus etiquetas y colores se leen de /config.
 */

function capitalize(text: string): string {
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function humanize(name: string): string {
  return capitalize(name.replace(/_/g, ' '));
}

export const NOT_MEASURED = 'sin medir';

/** Un tiempo null significa que no se midió: nunca se presenta como 0. */
export function formatMs(ms: number | null | undefined, locale = 'es'): string {
  if (ms === null || ms === undefined) return NOT_MEASURED;
  const seconds = ms >= 1000;
  const text = (seconds ? ms / 1000 : ms).toLocaleString(locale, {
    minimumFractionDigits: 1,
    maximumFractionDigits: 1,
  });
  return `${text} ${seconds ? 's' : 'ms'}`;
}

export type ParticleStyle = 'polen' | 'brasas' | 'chispas' | 'polvo' | 'none';

export interface MoodPresentation {
  label: string;
  tagline: string;
  /** Filtro CSS orientativo para la tarjeta. No es el render del motor. */
  filter: string;
  /** Veladura sobre la miniatura. */
  wash: string;
  particle: ParticleStyle;
  particleColor: string;
}

const MOODS: Readonly<Record<string, MoodPresentation>> = {
  calma: {
    label: 'Calma',
    tagline: 'Lento y respirado, con polen flotando.',
    filter: 'saturate(.95) brightness(1.03)',
    wash: 'linear-gradient(160deg, rgba(150,220,200,.2), rgba(20,40,60,.35))',
    particle: 'polen',
    particleColor: '#f8f3c4',
  },
  epico: {
    label: 'Épico',
    tagline: 'Rápido y contrastado, con brasas al viento.',
    filter: 'saturate(1.15) contrast(1.2)',
    wash: 'linear-gradient(160deg, rgba(30,50,110,.35), rgba(255,80,30,.28))',
    particle: 'brasas',
    particleColor: '#ff8a4c',
  },
  magico: {
    label: 'Mágico',
    tagline: 'Color vivo y chispas que centellean.',
    filter: 'saturate(1.25) contrast(1.05) hue-rotate(-8deg)',
    wash: 'linear-gradient(160deg, rgba(120,60,255,.3), rgba(40,200,255,.2))',
    particle: 'chispas',
    particleColor: '#d7c7ff',
  },
  nostalgico: {
    label: 'Nostálgico',
    tagline: 'Tonos cálidos y polvo en la luz.',
    filter: 'saturate(.7) sepia(.3) contrast(.95)',
    wash: 'linear-gradient(160deg, rgba(240,180,90,.22), rgba(70,40,20,.4))',
    particle: 'polvo',
    particleColor: '#f0dcb4',
  },
};

export function moodPresentation(name: string): MoodPresentation {
  return (
    MOODS[name] ?? {
      label: humanize(name),
      tagline: '',
      filter: 'none',
      wash: 'none',
      particle: 'none',
      particleColor: '#ffffff',
    }
  );
}
