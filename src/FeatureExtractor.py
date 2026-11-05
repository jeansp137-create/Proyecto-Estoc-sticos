"""Procesamiento geométrico: del trazo continuo a símbolos discretos y viceversa.

Pipeline (igual en 2D y 3D):
    trazos -> ruta plana por trazo -> remuestreo -> código de cadena extendido

Alfabeto de observaciones (K = 18 símbolos):
    0 .. 15 : sectores angulares del código de cadena (cada uno de 22.5 grados).
    16      : salto en el aire (pen-up: el usuario levantó el lápiz y empezó
              un nuevo trazo), seguido por el sector angular hacia el nuevo trazo.
    17      : punto corto (dot, como el punto de la 'i' o la 'j').

* 2D: los puntos (x, y) llegan en coordenadas de pantalla (x derecha, y abajo).
  Se invierte el eje y para medir ángulos trigonométricos estándar (y arriba).
* 3D: los puntos (x, y, z) sobre la esfera unitaria se transforman a coordenadas
  esféricas (theta, phi) y se evalúan las variaciones (d_theta, d_phi) sobre la
  superficie esférica (sección 3.4 de la guía):
      dx = d_phi * sin(theta),   dy = -d_theta
* Reconstrucción / Proyección futura (sección 3.5 de la guía):
  A partir de los símbolos futuros {O_{t+1}, ..., O_T} predichos por el HMM de
  mayor probabilidad, reconstruye las coordenadas proyectadas en 2D o sobre la
  superficie de la esfera 3D desde la posición actual del cursor.

Sólo usa NumPy. No depende de la interfaz gráfica.
"""
import numpy as np


class FeatureExtractor:
    K_ANG = 16           # sectores angulares (22.5 grados c/u)
    SYM_PEN_UP = 16      # levantar el lápiz entre trazos
    SYM_DOT = 17         # punto (dot de 'i' / 'j')
    K = 18               # tamaño total del alfabeto de observaciones
    PASO_2D = 18.0       # paso en píxeles en el lienzo 2D
    PASO_DATASET = 25.0  # paso en unidades del dataset (lienzo 1000x1000)
    PASO_3D = 0.055      # paso angular en radianes sobre la esfera 3D

    # ------------------------------------------------------------ utilidades
    @staticmethod
    def _normalizar_trazos_2d(entrada):
        """Convierte cualquier representación de entrada 2D en una lista de trazos [(N_i, 2), ...]."""
        if entrada is None:
            return []
        if isinstance(entrada, np.ndarray):
            if entrada.ndim == 2 and len(entrada) > 0:
                return [entrada[:, :2].astype(float)]
            return []
        if not entrada:
            return []
        primero = entrada[0]
        # Lista de trazos: [[[x, y, ...], ...], ...]
        if isinstance(primero, (list, tuple, np.ndarray)) and len(primero) > 0 and isinstance(primero[0], (list, tuple, np.ndarray)):
            trazos = []
            for tr in entrada:
                if tr is not None and len(tr) > 0:
                    trazos.append(np.asarray([p[:2] for p in tr], dtype=float).reshape(-1, 2))
            return trazos
        # Un solo trazo plano: [(x, y), ...]
        return [np.asarray([p[:2] for p in entrada], dtype=float).reshape(-1, 2)]

    @staticmethod
    def _normalizar_trazos_3d(entrada):
        """Convierte cualquier representación de entrada 3D en una lista de trazos [(N_i, 3), ...]."""
        if entrada is None:
            return []
        if isinstance(entrada, np.ndarray):
            if entrada.ndim == 2 and len(entrada) > 0:
                return [entrada[:, :3].astype(float)]
            return []
        if not entrada:
            return []
        primero = entrada[0]
        if isinstance(primero, (list, tuple, np.ndarray)) and len(primero) > 0 and isinstance(primero[0], (list, tuple, np.ndarray)):
            trazos = []
            for tr in entrada:
                if tr is not None and len(tr) > 0:
                    trazos.append(np.asarray([p[:3] for p in tr], dtype=float).reshape(-1, 3))
            return trazos
        return [np.asarray([p[:3] for p in entrada], dtype=float).reshape(-1, 3)]

    @staticmethod
    def aplanar_muestra(muestra):
        """Devuelve todos los puntos concatenados (N, 2)."""
        trazos = FeatureExtractor._normalizar_trazos_2d(muestra)
        if not trazos:
            return np.zeros((0, 2))
        return np.vstack(trazos)

    def ruta_2d(self, puntos):
        """Convierte puntos (x, y) con y hacia abajo a ruta plana con y hacia arriba."""
        p = self.aplanar_muestra(puntos)
        if len(p) == 0:
            return p
        return p * np.array([1.0, -1.0])

    # ------------------------------------------------------------ 3D
    @staticmethod
    def esfericas(puntos, rot=None):
        """Transforma puntos (x, y, z) de la esfera a coordenadas angulares (theta, phi)."""
        p = np.asarray(puntos, dtype=float).reshape(-1, 3)
        if len(p) == 0:
            return np.zeros(0), np.zeros(0)
        p = p / np.maximum(1e-12, np.linalg.norm(p, axis=1, keepdims=True))
        if rot is not None:
            p = p @ np.asarray(rot, dtype=float).T
        theta = np.arccos(np.clip(p[:, 1], -1.0, 1.0))
        phi = np.arctan2(p[:, 0], p[:, 2])
        return theta, phi

    def _trazos_esfericos_a_2d(self, trazos_3d, rot=None):
        """Convierte una lista de trazos 3D sobre la esfera en rutas 2D tangentes (theta, phi)."""
        rutas = []
        theta0 = phi0 = None
        for tr in trazos_3d:
            theta, phi = self.esfericas(tr, rot=rot)
            if len(theta) == 0:
                continue
            if theta0 is None:
                theta0, phi0 = float(theta[0]), float(phi[0])
            d_phi = (phi - phi0 + np.pi) % (2 * np.pi) - np.pi
            x = d_phi * np.sin(0.5 * (theta + theta0))
            y = -(theta - theta0)
            rutas.append(np.stack([x, y], axis=1))
        return rutas

    @staticmethod
    def muestra_2d_a_esfera(muestra):
        """Proyecta una muestra 2D (uno o varios trazos) sobre la esfera unitaria (3D)."""
        trazos_2d = FeatureExtractor._normalizar_trazos_2d(muestra)
        if not trazos_2d:
            return []
        P_all = np.vstack(trazos_2d)
        if len(P_all) < 2:
            return []
        centro = 0.5 * (P_all.min(axis=0) + P_all.max(axis=0))
        escala = max(220.0, float((P_all.max(axis=0) - P_all.min(axis=0)).max()))
        trazos_3d = []
        for P in trazos_2d:
            uv = (P - centro) / (escala * 1.15)
            u, v = uv[:, 0], -uv[:, 1]
            rho = np.hypot(u, v)
            alfa = np.arctan2(v, u)
            sr, cr = np.sin(rho), np.cos(rho)
            S = np.stack([sr * np.cos(alfa), sr * np.sin(alfa), cr], axis=1)
            S /= np.maximum(1e-12, np.linalg.norm(S, axis=1, keepdims=True))
            trazos_3d.append(S)
        return trazos_3d

    # ------------------------------------------------------------ discretización
    @staticmethod
    def remuestrear(ruta, paso):
        """Remuestrea la polilínea a puntos equiespaciados cada `paso`."""
        ruta = np.asarray(ruta, dtype=float)
        if len(ruta) < 2:
            return ruta[:0]
        d = np.hypot(*np.diff(ruta, axis=0).T)
        ruta = ruta[np.concatenate([[True], d > 1e-6])]
        if len(ruta) < 2:
            return ruta[:0]
        d = np.hypot(*np.diff(ruta, axis=0).T)
        s = np.concatenate([[0.0], np.cumsum(d)])
        if s[-1] < paso * 0.5:
            return ruta[:0]
        t = np.arange(0.0, max(paso, s[-1]) + 1e-9, paso)
        if len(t) < 2:
            t = np.array([0.0, s[-1]])
        return np.stack([np.interp(t, s, ruta[:, 0]), np.interp(t, s, ruta[:, 1])], axis=1)

    def _sector_angular(self, dx, dy):
        ang = np.arctan2(dy, dx)
        sector = 2 * np.pi / self.K_ANG
        return (np.floor(((ang + sector / 2) % (2 * np.pi)) / sector).astype(int)) % self.K_ANG

    def codigo_cadena(self, ruta):
        """Cuantiza desplazamientos (dx, dy) en K_ANG sectores angulares [0 .. 15]."""
        if len(ruta) < 2:
            return np.zeros(0, dtype=int)
        d = np.diff(ruta, axis=0)
        return self._sector_angular(d[:, 0], d[:, 1])

    def _secuencia_desde_rutas(self, rutas_2d_y_arriba, paso):
        """Convierte una lista de rutas (con y hacia arriba) en símbolos [0..17]."""
        seq = []
        prev_fin = None
        sector = 2 * np.pi / self.K_ANG
        for P in rutas_2d_y_arriba:
            if len(P) == 0:
                continue
            if prev_fin is not None:
                salto = P[0] - prev_fin
                ang_s = float(np.arctan2(salto[1], salto[0]))
                sym_s = int(np.floor(((ang_s + sector / 2) % (2 * np.pi)) / sector)) % self.K_ANG
                seq.append(self.SYM_PEN_UP)
                seq.append(sym_s)
            if len(P) == 1:
                seq.append(self.SYM_DOT)
            else:
                d = np.hypot(*np.diff(P, axis=0).T)
                if d.sum() < paso * 0.5:
                    seq.append(self.SYM_DOT)
                else:
                    R = self.remuestrear(P, paso)
                    if len(R) >= 2:
                        seq.extend(self.codigo_cadena(R).tolist())
                    else:
                        seq.append(self.SYM_DOT)
            prev_fin = P[-1]
        return np.asarray(seq, dtype=int)

    def simbolos_2d(self, entrada, paso=None):
        paso = self.PASO_2D if paso is None else paso
        trazos = self._normalizar_trazos_2d(entrada)
        rutas = [tr * np.array([1.0, -1.0]) for tr in trazos]
        return self._secuencia_desde_rutas(rutas, paso)

    def simbolos_dataset_2d(self, muestra):
        return self.simbolos_2d(muestra, paso=self.PASO_DATASET)

    def simbolos_3d(self, entrada, rot=None, paso=None):
        paso = self.PASO_3D if paso is None else paso
        trazos_3d = self._normalizar_trazos_3d(entrada)
        rutas = self._trazos_esfericos_a_2d(trazos_3d, rot=rot)
        return self._secuencia_desde_rutas(rutas, paso)

    # ------------------------------------------------------------ proyección inversa
    def simbolo_a_direccion(self, k):
        """Devuelve el vector unitario (cos, sin) del sector k (con y hacia arriba)."""
        a = 2 * np.pi * (int(k) % self.K_ANG) / self.K_ANG
        return float(np.cos(a)), float(np.sin(a))

    def proyectar_2d(self, punto_actual, simbolos_futuros, paso=None):
        """Reconstruye puntos 2D [(x, y), ...] desde `punto_actual` siguiendo los símbolos."""
        if not simbolos_futuros:
            return []
        paso = self.PASO_2D if paso is None else paso
        x, y = float(punto_actual[0]), float(punto_actual[1])
        pts = [(x, y)]
        salto = False
        for k in simbolos_futuros:
            k = int(k)
            if k == self.SYM_PEN_UP:
                salto = True
                continue
            if k == self.SYM_DOT:
                pts.append((x, y))
                continue
            ux, uy = self.simbolo_a_direccion(k)
            mult = 2.2 if salto else 1.0
            salto = False
            x += paso * mult * ux
            y -= paso * mult * uy  # en pantalla y crece hacia abajo
            pts.append((x, y))
        return pts

    def proyectar_3d(self, punto_actual, simbolos_futuros, rot=None, paso=None):
        """Reconstruye puntos 3D [(x, y, z), ...] sobre la esfera desde `punto_actual`."""
        if not simbolos_futuros:
            return []
        paso = self.PASO_3D if paso is None else paso
        R = np.eye(3) if rot is None else np.asarray(rot, dtype=float)
        p = np.asarray(punto_actual, dtype=float)
        p = p / max(1e-12, float(np.linalg.norm(p)))
        pv = R @ p
        theta = float(np.arccos(np.clip(pv[1], -1.0, 1.0)))
        phi = float(np.arctan2(pv[0], pv[2]))
        pts = [tuple(float(v) for v in p)]
        salto = False
        for k in simbolos_futuros:
            k = int(k)
            if k == self.SYM_PEN_UP:
                salto = True
                continue
            if k == self.SYM_DOT:
                continue
            ux, uy = self.simbolo_a_direccion(k)
            mult = 2.2 if salto else 1.0
            salto = False
            d_theta = -paso * mult * uy
            st = max(0.20, float(np.sin(theta)))
            d_phi = (paso * mult * ux) / st
            theta = float(np.clip(theta + d_theta, 0.08, np.pi - 0.08))
            phi = float(phi + d_phi)
            qv = np.array([
                np.sin(theta) * np.sin(phi),
                np.cos(theta),
                np.sin(theta) * np.cos(phi),
            ])
            qw = R.T @ qv
            qw /= max(1e-12, float(np.linalg.norm(qw)))
            pts.append(tuple(float(v) for v in qw))
        return pts
