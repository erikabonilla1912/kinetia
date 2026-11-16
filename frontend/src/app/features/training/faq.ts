import { ChangeDetectionStrategy, Component, input } from '@angular/core';
import { formatNumber } from './training-summary';

/**
 * Preguntas frecuentes para visitantes. Cada respuesta describe el código desplegado:
 * - privacidad: backend/app/services/job_manager.py (borra la entrada al terminar; RESULT_TTL_HOURS=96)
 * - animación: backend/animation_engine/effects y la decisión 13 de PROGRESO.md (sin parallax)
 * Las cifras del modelo llegan por entradas, de /model/metrics y del history.json.
 */
@Component({
  selector: 'app-faq',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <section class="glass mt-4 p-4 sm:p-5" data-testid="faq">
      <h2 class="text-2xl font-bold">Preguntas frecuentes</h2>
      <div class="mt-2 divide-y divide-white/10">
        @for (item of items(); track item.q) {
          <details class="group py-1" data-testid="faq-item">
            <summary class="flex min-h-11 cursor-pointer items-center justify-between gap-3 font-semibold text-paper">
              {{ item.q }}
              <span class="text-xray transition-transform group-open:rotate-45" aria-hidden="true">+</span>
            </summary>
            <p class="pb-3 text-sm text-mist">{{ item.a }}</p>
          </details>
        }
      </div>
    </section>
  `,
})
export class Faq {
  readonly testMiou = input<number | null>(null);
  readonly epochs = input<number>(0);

  protected items(): { q: string; a: string }[] {
    const miou = this.testMiou();
    const precision =
      miou === null
        ? 'La cifra oficial está en la página Modelo.'
        : `En 2268 fotos que nunca vio al entrenar logra un mIoU de ${formatNumber(miou, 4)}: en promedio, la zona que marca coincide en un ${Math.round(miou * 100)} % con la que marcó una persona.`;
    return [
      {
        q: '¿Qué hace KINETIA?',
        a: 'Subes una foto o un video corto. Un modelo reconoce qué hay en cada píxel (cielo, agua, vegetación, fuego o luz, personas, animales o fondo) y cada zona se anima a su manera: las nubes se desplazan, el agua ondula, las hojas se mecen y el fuego parpadea, con partículas según el ambiente que elijas.',
      },
      {
        q: '¿Qué modelo usa?',
        a: 'Una red U-Net con encoder MobileNetV3 Large, de 6,69 millones de parámetros. La entrenamos nosotros con 10 583 fotos de COCO-Stuff y BoWFire.',
      },
      {
        q: '¿Usaron un modelo ya entrenado?',
        a: `No. La red empezó con pesos aleatorios y aprendió solo de nuestras fotos durante ${this.epochs()} épocas en Google Colab. Tampoco usamos estadísticas de ImageNet: la normalización se calculó con nuestras propias fotos.`,
      },
      {
        q: '¿Qué tan preciso es?',
        a: `${precision} Reconoce mejor el fondo, el cielo y el agua; la clase más difícil es fuego y luz, porque tenía pocas fotos.`,
      },
      {
        q: '¿Por qué a veces se equivoca?',
        a: 'Aprendió de fotografías reales, así que le cuestan las fotos de noche, los dibujos y las zonas parecidas entre sí: unas montañas azuladas pueden salir como agua y animarse con ondas.',
      },
      {
        q: '¿Por qué las personas y los animales no se mueven?',
        a: 'Es una decisión de diseño: el sujeto queda quieto y se anima lo que lo rodea, como en un cinemagraph. Mover un cuerpo de forma creíble exigiría un modelo generativo y una GPU, y este proyecto corre en la CPU de un portátil.',
      },
      {
        q: '¿Dónde se procesa mi foto?',
        a: 'En este portátil, sin GPU y sin servicios en la nube. La app corre en Kubernetes y el modelo se ejecuta con ONNX Runtime en la CPU.',
      },
      {
        q: '¿Qué pasa con mi foto?',
        a: 'La foto original se borra en cuanto termina el proceso, salga bien o mal. El video se borra solo a las 96 horas. Solo aparece en el muro de la Expo si das permiso, y el equipo puede retirarlo. Los registros del servidor no guardan imágenes.',
      },
    ];
  }
}
