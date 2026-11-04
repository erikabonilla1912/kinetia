import {
  AfterViewInit,
  ChangeDetectionStrategy,
  Component,
  ElementRef,
  OnDestroy,
  viewChild,
} from '@angular/core';
import gsap from 'gsap';
import { prefersReducedMotion } from '../../core/motion';

const WORDS_TOP = ['Dale', 'vida'];
const WORDS_BOTTOM = ['a', 'tus', 'imágenes'];

/**
 * Titular con una idea: las letras de "vida" respiran. Es decir, el título hace
 * lo que el producto promete. Sin movimiento reducido queda quieto y legible.
 */
@Component({
  selector: 'app-hero-title',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <h1
      #title
      class="hero font-display font-extrabold text-paper"
      aria-label="Dale vida a tus imágenes"
    >
      <span class="line" aria-hidden="true">
        @for (word of top; track word) {
          <span class="word" [class.alive]="word === 'vida'" data-word>
            @for (letter of letters(word); track $index) {
              <span class="letter inline-block" [class.text-pulse]="word === 'vida'">{{ letter }}</span>
            }
          </span>
        }
      </span>
      <span class="line" aria-hidden="true">
        @for (word of bottom; track word) {
          <span class="word" data-word>{{ word }}</span>
        }
      </span>
    </h1>
  `,
  styles: `
    .hero {
      font-size: 19vw;
      line-height: 0.86;
      letter-spacing: -0.045em;
    }
    @media (min-width: 1024px) {
      .hero {
        font-size: min(14.5vw, 30dvh);
      }
    }
    .line {
      display: block;
    }
    .line + .line {
      font-size: 0.52em;
      line-height: 1;
      letter-spacing: -0.035em;
      margin-top: 0.14em;
      color: var(--color-mist);
    }
    .word {
      display: inline-block;
      margin-right: 0.22em;
    }
    .word:last-child {
      margin-right: 0;
    }
  `,
})
export class HeroTitle implements AfterViewInit, OnDestroy {
  protected readonly top = WORDS_TOP;
  protected readonly bottom = WORDS_BOTTOM;

  private readonly title = viewChild.required<ElementRef<HTMLElement>>('title');
  private context: gsap.Context | null = null;

  protected letters(word: string): string[] {
    return [...word];
  }

  ngAfterViewInit(): void {
    if (prefersReducedMotion()) return;
    const root = this.title().nativeElement;
    this.context = gsap.context(() => {
      gsap.from(root.querySelectorAll('[data-word]'), {
        yPercent: 40,
        opacity: 0,
        duration: 1,
        stagger: 0.08,
        ease: 'power4.out',
        clearProps: 'transform,opacity',
      });
      // Una onda lenta recorre "vida", como agua: es lo único que se mueve siempre.
      gsap.to(root.querySelectorAll('.alive .letter'), {
        yPercent: -7,
        rotate: -1.5,
        duration: 1.6,
        ease: 'sine.inOut',
        repeat: -1,
        yoyo: true,
        delay: 1,
        stagger: { each: 0.22, repeat: -1, yoyo: true },
      });
    }, root);
  }

  ngOnDestroy(): void {
    this.context?.revert();
  }
}
