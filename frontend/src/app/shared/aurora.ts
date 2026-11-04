import {
  AfterViewInit,
  ChangeDetectionStrategy,
  Component,
  ElementRef,
  OnDestroy,
  inject,
} from '@angular/core';
import gsap from 'gsap';
import { prefersReducedMotion } from '../core/motion';

/**
 * Luz que deriva lentamente detrás del vidrio. Solo anima `transform`, que no repinta,
 * y se usa únicamente donde no hay videos compitiendo por la GPU.
 */
@Component({
  selector: 'app-aurora',
  changeDetection: ChangeDetectionStrategy.OnPush,
  host: {
    class: 'pointer-events-none fixed inset-0 block overflow-hidden',
    style: 'z-index: -1',
    'aria-hidden': 'true',
  },
  template: `
    <div class="orb" style="left: 52%; top: -30%; --c: rgb(180 140 255 / 0.34)"></div>
    <div class="orb" style="left: -22%; top: 18%; --c: rgb(107 231 240 / 0.22)"></div>
    <div class="orb" style="left: 30%; top: 46%; --c: rgb(255 106 61 / 0.26)"></div>
  `,
  styles: `
    .orb {
      position: absolute;
      width: 90vmax;
      height: 70vmax;
      background: radial-gradient(closest-side, var(--c), transparent);
      will-change: transform;
    }
  `,
})
export class Aurora implements AfterViewInit, OnDestroy {
  private readonly host = inject<ElementRef<HTMLElement>>(ElementRef);
  private context: gsap.Context | null = null;

  ngAfterViewInit(): void {
    if (prefersReducedMotion()) return;
    const root = this.host.nativeElement;
    this.context = gsap.context(() => {
      root.querySelectorAll('.orb').forEach((orb, index) => {
        gsap.to(orb, {
          xPercent: index % 2 ? 22 : -18,
          yPercent: index === 1 ? -16 : 14,
          scale: 1.12,
          duration: 13 + index * 4,
          ease: 'sine.inOut',
          repeat: -1,
          yoyo: true,
        });
      });
    }, root);
  }

  ngOnDestroy(): void {
    this.context?.revert();
  }
}
