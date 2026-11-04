import { ChangeDetectionStrategy, Component, computed, input } from '@angular/core';
import { encode } from 'uqr';

/** Módulos de zona de silencio que exige el estándar QR alrededor del código. */
const QUIET_ZONE = 4;

/** QR en SVG: escala sin perder nitidez en la pantalla grande. */
@Component({
  selector: 'app-qr-code',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <svg
      role="img"
      class="block h-full w-full"
      shape-rendering="crispEdges"
      [attr.viewBox]="viewBox()"
      [attr.aria-label]="'Código QR que abre ' + value()"
    >
      <rect [attr.width]="size()" [attr.height]="size()" fill="#ffffff" />
      <path [attr.d]="path()" fill="#000000" />
    </svg>
  `,
})
export class QrCode {
  readonly value = input.required<string>();

  private readonly matrix = computed(() => encode(this.value(), { ecc: 'M', border: 0 }).data);
  protected readonly size = computed(() => this.matrix().length + QUIET_ZONE * 2);
  protected readonly viewBox = computed(() => `0 0 ${this.size()} ${this.size()}`);
  protected readonly path = computed(() => {
    let d = '';
    this.matrix().forEach((row, y) =>
      row.forEach((dark, x) => {
        if (dark) d += `M${x + QUIET_ZONE} ${y + QUIET_ZONE}h1v1h-1z`;
      }),
    );
    return d;
  });
}
