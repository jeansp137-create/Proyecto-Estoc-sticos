"""Procesamiento geométrico: del trazo continuo a símbolos discretos y viceversa.

Pipeline (igual en 2D y 3D):
    trazo -> ruta plana -> remuestreo a paso fijo -> código de cadena (K=16 sectores)

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
    K = 16               # sectores angulares del código de cadena (22.5 grados c/u)
    PASO_2D = 18.0       # paso en píxeles en el lienzo 2D
    PASO_DATASET = 25.0  # paso en unidades del dataset (lienzo 1000x1000)
    PASO_3D = 0.055      # paso angular en radianes sobre la esfera 3D

    # ------------------------------------------------------------ 2D
    @staticmethod
    def aplanar_muestra(muestra):
        """Acepta [[x, y], ...] o múltiples trazos [[[x, y, t], ...], ...] del dataset."""
        if not muestra:
            return np.zeros((0, 2))
        primero = muestra[0]
        if isinstance(primero, (list, tuple)) and len(primero) > 0 and isinstance(primero[0], (list, tuple)):
            pts = [p[:2] for trazo in muestra for p in trazo]
        else:
            pts = [p[:2] for p in muestra]
        if not pts:
            return np.zeros((0, 2))
        return np.asarray(pts, dtype=float).reshape(-1, 2)

    def ruta_2d(self, puntos):
        """Convierte puntos (x, y) con y hacia abajo a ruta plana con y hacia arriba."""
        p = self.aplanar_muestra(puntos)
        if len(p) == 0:
            return p
        return p * np.array([1.0, -1.0])

    # ------------------------------------------------------------ 3D
    @staticmethod
    def esfericas(puntos, rot=None):
        """Transforma puntos (x, y, z) de la esfera a coordenadas angulares (theta, phi).

        Si se pasa `rot` (matriz 3x3 de vista de la cámara), primero rota los
        puntos al marco de la cámara (x derecha, y arriba, z hacia el observador)
        para que el trazo sea invariante al giro de la esfera.
        theta: colatitud desde el polo +y (arriba), en [0, pi].
        phi:   longitud en el plano xz desde +z, en [-pi, pi].
        """
        p = np.asarray(puntos, dtype=float).reshape(-1, 3)
        if len(p) == 0:
            return np.zeros(0), np.zeros(0)
        p = p / np.maximum(1e-12, np.linalg.norm(p, axis=1, keepdims=True))
        if rot is not None:
            p = p @ np.asarray(rot, dtype=float).T
        theta = np.arccos(np.clip(p[:, 1], -1.0, 1.0))
        phi = np.arctan2(p[:, 0], p[:, 2])
        return theta, phi

    def ruta_3d(self, puntos, rot=None):
        """Integra las variaciones esféricas (d_theta, d_phi) en una ruta 2D tangente."""
        theta, phi = self.esfericas(puntos, rot=rot)
        if len(theta) < 2:
            return np.zeros((len(theta), 2))
        d_theta = np.diff(theta)
        d_phi = (np.diff(phi) + np.pi) % (2 * np.pi) - np.pi
        theta_m = 0.5 * (theta[1:] + theta[:-1])
        paso = np.stack([d_phi * np.sin(theta_m), -d_theta], axis=1)
        return np.vstack([[0.0, 0.0], np.cumsum(paso, axis=0)])

    @staticmethod
    def muestra_2d_a_esfera(muestra):
        """Proyecta una muestra 2D del dataset sobre la esfera unitaria (3D)."""
        P = FeatureExtractor.aplanar_muestra(muestra)
        if len(P) < 2:
            return np.zeros((0, 3))
        centro = 0.5 * (P.min(axis=0) + P.max(axis=0))
        escala = max(220.0, float((P.max(axis=0) - P.min(axis=0)).max()))
        uv = (P - centro) / (escala * 1.15)
        u, v = uv[:, 0], -uv[:, 1]
        rho = np.hypot(u, v)
        alfa = np.arctan2(v, u)
        # Mapeo exponencial desde el centro frontal (0, 0, 1)
        sr, cr = np.sin(rho), np.cos(rho)
        x = sr * np.cos(alfa)
        y = sr * np.sin(alfa)
        z = cr
        S = np.stack([x, y, z], axis=1)
        return S / np.maximum(1e-12, np.linalg.norm(S, axis=1, keepdims=True))

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
        if s[-1] < paso:
            return ruta[:0]
        t = np.arange(0.0, s[-1] + 1e-9, paso)
        return np.stack([np.interp(t, s, ruta[:, 0]), np.interp(t, s, ruta[:, 1])], axis=1)

    def codigo_cadena(self, ruta):
        """Cuantiza desplazamientos (dx, dy) en K sectores angulares [0 .. K-1]."""
        if len(ruta) < 2:
            return np.zeros(0, dtype=int)
        d = np.diff(ruta, axis=0)
        ang = np.arctan2(d[:, 1], d[:, 0])
        sector = 2 * np.pi / self.K
        return (np.floor(((ang + sector / 2) % (2 * np.pi)) / sector).astype(int)) % self.K

    def simbolos_2d(self, puntos, paso=None):
        paso = self.PASO_2D if paso is None else paso
        return self.codigo_cadena(self.remuestrear(self.ruta_2d(puntos), paso))

    def simbolos_dataset_2d(self, muestra):
        return self.simbolos_2d(muestra, paso=self.PASO_DATASET)

    def simbolos_3d(self, puntos, rot=None, paso=None):
        paso = self.PASO_3D if paso is None else paso
        return self.codigo_cadena(self.remuestrear(self.ruta_3d(puntos, rot=rot), paso))

    # ------------------------------------------------------------ proyección inversa
    def simbolo_a_direccion(self, k):
        """Devuelve el vector unitario (cos, sin) del sector k (con y hacia arriba)."""
        a = 2 * np.pi * int(k) / self.K
        return float(np.cos(a)), float(np.sin(a))

    def proyectar_2d(self, punto_actual, simbolos_futuros, paso=None):
        """Reconstruye puntos 2D [(x, y), ...] desde `punto_actual` siguiendo los símbolos."""
        if not simbolos_futuros:
            return []
        paso = self.PASO_2D if paso is None else paso
        x, y = float(punto_actual[0]), float(punto_actual[1])
        pts = [(x, y)]
        for k in simbolos_futuros:
            ux, uy = self.simbolo_a_direccion(k)
            x += paso * ux
            y -= paso * uy  # en pantalla y crece hacia abajo
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
        # Pasamos al marco de la vista para avanzar en (theta, phi)
        pv = R @ p
        theta = float(np.arccos(np.clip(pv[1], -1.0, 1.0)))
        phi = float(np.arctan2(pv[0], pv[2]))
        pts = [tuple(float(v) for v in p)]
        for k in simbolos_futuros:
            ux, uy = self.simbolo_a_direccion(k)
            d_theta = -paso * uy
            st = max(0.20, float(np.sin(theta)))
            d_phi = (paso * ux) / st
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
