import { ComponentFixture, TestBed } from '@angular/core/testing';
import { Comparator, nextSliderValue } from './comparator';

const PIXEL = 'data:image/gif;base64,R0lGODlhAQABAAAAACw=';

describe('nextSliderValue', () => {
  it('mueve con flechas, página, inicio y fin, acotado a 0-100', () => {
    expect(nextSliderValue(50, 'ArrowRight')).toBe(52);
    expect(nextSliderValue(50, 'ArrowUp')).toBe(52);
    expect(nextSliderValue(50, 'ArrowLeft')).toBe(48);
    expect(nextSliderValue(50, 'ArrowDown')).toBe(48);
    expect(nextSliderValue(50, 'PageUp')).toBe(60);
    expect(nextSliderValue(50, 'PageDown')).toBe(40);
    expect(nextSliderValue(50, 'Home')).toBe(0);
    expect(nextSliderValue(50, 'End')).toBe(100);
    expect(nextSliderValue(99, 'ArrowRight')).toBe(100);
    expect(nextSliderValue(1, 'ArrowLeft')).toBe(0);
    expect(nextSliderValue(50, 'a')).toBeNull();
  });
});

describe('Comparator', () => {
  let fixture: ComponentFixture<Comparator>;
  const root = () => fixture.nativeElement as HTMLElement;
  const q = (id: string) => root().querySelector<HTMLElement>(`[data-testid="${id}"]`)!;

  beforeEach(async () => {
    // Sin barrido inicial: la prueba controla la posición.
    spyOn(window, 'matchMedia').and.returnValue({ matches: true } as MediaQueryList);
    fixture = TestBed.createComponent(Comparator);
    fixture.componentRef.setInput('beforeUrl', PIXEL);
    fixture.componentRef.setInput('afterUrl', `${PIXEL}#.gif`);
    root().style.cssText = 'display:block;position:relative;width:400px;height:200px';
    await fixture.whenStable();
  });

  it('expone un slider accesible', () => {
    const handle = q('comparator-handle');
    expect(handle.getAttribute('role')).toBe('slider');
    expect(handle.getAttribute('tabindex')).toBe('0');
    expect(handle.getAttribute('aria-valuemin')).toBe('0');
    expect(handle.getAttribute('aria-valuemax')).toBe('100');
    expect(handle.getAttribute('aria-valuenow')).toBe('50');
    expect(handle.getAttribute('aria-label')).toBeTruthy();
  });

  it('el teclado mueve el corte y actualiza aria-valuenow', async () => {
    const handle = q('comparator-handle');
    const press = async (key: string) => {
      const event = new KeyboardEvent('keydown', { key, cancelable: true, bubbles: true });
      handle.dispatchEvent(event);
      await fixture.whenStable();
      return event;
    };
    const event = await press('ArrowRight');
    expect(event.defaultPrevented).toBeTrue();
    expect(handle.getAttribute('aria-valuenow')).toBe('52');
    await press('Home');
    expect(handle.getAttribute('aria-valuenow')).toBe('0');
    await press('End');
    expect(handle.getAttribute('aria-valuenow')).toBe('100');
    const ignored = await press('x');
    expect(ignored.defaultPrevented).toBeFalse();
  });

  it('recorta la capa "antes" según la posición', async () => {
    fixture.componentInstance.position.set(30);
    await fixture.whenStable();
    expect(q('before-layer').style.clipPath).toBe('inset(0px 70% 0px 0px)');
    expect(q('comparator-handle').style.left).toBe('30%');
  });

  it('arrastrar mueve el corte según el puntero, acotado a los bordes', async () => {
    const area = q('comparator-area');
    const rect = area.getBoundingClientRect();
    const fire = (type: string, fraction: number) =>
      area.dispatchEvent(
        new PointerEvent(type, { clientX: rect.left + rect.width * fraction, bubbles: true, pointerId: 1 }),
      );
    fire('pointerdown', 0.25);
    expect(fixture.componentInstance.position()).toBeCloseTo(25, 0);
    fire('pointermove', 0.75);
    expect(fixture.componentInstance.position()).toBeCloseTo(75, 0);
    fire('pointermove', 5);
    expect(fixture.componentInstance.position()).toBe(100);
    fire('pointerup', 0.1);
    fire('pointermove', 0.1);
    expect(fixture.componentInstance.position()).toBe(100);
  });
});
