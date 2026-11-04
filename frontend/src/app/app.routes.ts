import { Routes } from '@angular/router';

export const routes: Routes = [
  {
    path: '',
    pathMatch: 'full',
    title: 'KINETIA · Dale vida a tus imágenes',
    loadComponent: () => import('./features/studio/studio.page').then((m) => m.StudioPage),
  },
  {
    path: 'modelo',
    title: 'KINETIA · Métricas del modelo',
    loadComponent: () => import('./features/metrics/metrics.page').then((m) => m.MetricsPage),
  },
  {
    path: 'expo/subir',
    title: 'KINETIA · Anima tu foto',
    data: { chrome: false },
    loadComponent: () => import('./features/expo/expo-upload.page').then((m) => m.ExpoUploadPage),
  },
  {
    path: 'expo/pantalla',
    title: 'KINETIA · Muro',
    data: { chrome: false },
    loadComponent: () => import('./features/expo/expo-wall.page').then((m) => m.ExpoWallPage),
  },
  { path: '**', redirectTo: '' },
];
