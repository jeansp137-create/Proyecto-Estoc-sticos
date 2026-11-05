"""Interfaz gráfica del Reconocedor HMM 2D/3D (PyQt5).

Estilo minimalista con barra lateral: tipografía recta (Segoe UI), superficies
blancas, líneas finas y un único acento verde. Las letras de referencia se
dibujan como trazos manuscritos (ver Glifos.py) y alimentan la guía de trazado.
(Estilos anteriores respaldados en GUI_claro.py, GUI_brutal.py y GUI_minimal1.py.)

Navegación: barra lateral + QStackedWidget
    0 Inicio · 1 Reconocedor 2D · 2 Reconocedor 3D · 3 Guía de letras
La GUI sólo captura el trazo y muestra resultados; la lógica estocástica vive
en HMMEngine.py y la extracción de características en FeatureExtractor.py.
Los puntos marcados con "HOOK" son donde se conectará el motor HMM.
"""
import json
import os
import sys

import numpy as np
from PyQt5.QtCore import Qt, QPointF, QRectF, QTimer, pyqtSignal
from PyQt5.QtGui import (
    QColor, QFont, QPainter, QPainterPath, QPen, QRadialGradient, QVector3D,
)
from PyQt5.QtWidgets import (
    QApplication, QFrame, QGridLayout, QHBoxLayout, QLabel, QMainWindow,
    QPushButton, QScrollArea, QSizePolicy, QStackedWidget, QVBoxLayout, QWidget,
)

import Glifos
from FeatureExtractor import FeatureExtractor
from HMMEngine import HMMEngine, posteriores_desde_modelos

RAIZ_PROYECTO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_MODELOS = os.path.join(RAIZ_PROYECTO, "models")


def cargar_modelos_hmm(modo):
    """Carga models/modelos_<modo>.json; si no existe aún, lo entrena desde data/."""
    ruta = os.path.join(DIR_MODELOS, f"modelos_{modo}.json")
    if not os.path.exists(ruta):
        try:
            sys.path.insert(0, os.path.join(RAIZ_PROYECTO, "scripts"))
            import entrenar_modelos as em
            ds = em.cargar_dataset()
            seqs = em.preparar_secuencias(ds, modo=modo)
            if seqs:
                em.guardar_modelos(em.entrenar_modelos(seqs), modo)
        except Exception:
            pass
    if not os.path.exists(ruta):
        return {}
    with open(ruta, encoding="utf-8") as f:
        datos = json.load(f)
    return {c: HMMEngine.desde_dict(d) for c, d in datos.get("letras", {}).items()}

try:
    import pyqtgraph.opengl as gl
    HAS_GL = True
except Exception:  # PyOpenGL ausente: se usa la esfera de respaldo (QPainter)
    HAS_GL = False

TITULO = "Reconocedor HMM 2D/3D"
ALFABETO = list("abcdefghijklmnñopqrstuvwxyz")  # 27 letras

# ---------------------------------------------------------------- tema
BG = "#ffffff"
SIDE = "#f4f4f2"
SIDE_HOVER = "#e9e9e6"
INK = "#1a1a1a"
MUTED = "#6b6b6b"
LINE = "#e3e3e0"
TRACK = "#ececea"
ACCENT = "#0a7d5a"
ACCENT_SOFT = "#e2f3ec"

FAMILY = "Segoe UI"

PAGE_HOME, PAGE_2D, PAGE_3D, PAGE_GUIA = range(4)


def font(px, semibold=False):
    f = QFont(FAMILY)
    f.setPixelSize(px)
    f.setWeight(QFont.DemiBold if semibold else QFont.Normal)
    return f


STYLE = f"""
QMainWindow {{ background: {BG}; }}
QStackedWidget {{ background: {BG}; }}
QWidget {{ color: {INK}; font-family: '{FAMILY}'; font-size: 16px; }}
QLabel {{ background: transparent; }}
QLabel#muted {{ color: {MUTED}; font-size: 16px; }}
QPushButton {{ background: {BG}; border: 1px solid #cfcfcb; border-radius: 8px;
              color: {INK}; padding: 9px 20px; font-size: 16px; }}
QPushButton:hover {{ background: {ACCENT_SOFT}; border-color: {ACCENT}; color: {ACCENT}; }}
QPushButton:pressed {{ background: #cfe9de; }}
QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 6px; }}
QScrollBar::handle:vertical {{ background: #d4d4d0; min-height: 30px; border-radius: 3px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
"""


def regla():
    f = QFrame()
    f.setFixedHeight(1)
    f.setStyleSheet(f"background: {LINE};")
    return f


def etiqueta(texto, px, semibold=False, muted=False):
    l = QLabel(texto)
    l.setFont(font(px, semibold))
    if muted:
        l.setStyleSheet(f"color: {MUTED};")
    return l


# ---------------------------------------------------------------- glifos
def dibujar_glifo(p, letra, rect, color, ancho=2.0):
    """Dibuja la letra manuscrita centrada en rect (QRectF)."""
    s = min(rect.width() / Glifos.ANCHO, rect.height() / Glifos.ALTO)
    ox = rect.center().x() - Glifos.ANCHO * s / 2
    oy = rect.center().y() - Glifos.ALTO * s / 2
    p.save()
    p.setRenderHint(QPainter.Antialiasing)
    p.translate(ox, oy)
    p.scale(s, s)
    pen = QPen(QColor(color), ancho, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
    pen.setCosmetic(True)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    for path in Glifos.trazos(letra):
        p.drawPath(path)
    p.restore()


class GlyphWidget(QWidget):
    """Una letra manuscrita grande (la más probable)."""

    def __init__(self, color=ACCENT, ancho=6):
        super().__init__()
        self.letra, self.color, self.ancho = None, color, ancho

    def set_letter(self, letra):
        self.letra = letra
        self.update()

    def paintEvent(self, _):
        if not self.letra:
            return
        p = QPainter(self)
        dibujar_glifo(p, self.letra, QRectF(self.rect()).adjusted(8, 8, -8, -8),
                      self.color, self.ancho)


# ---------------------------------------------------------------- 2D
class DrawingCanvas(QWidget):
    """Lienzo 2D: hoja blanca, trazo negro y proyección punteada verde."""

    strokeChanged = pyqtSignal(list)  # lista de (x, y) del trazo actual

    def __init__(self):
        super().__init__()
        self.setMinimumSize(420, 380)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setCursor(Qt.CrossCursor)
        self.points = []
        self.projection = []
        self._drawing = False

    def clear(self):
        self.points, self.projection = [], []
        self.update()
        self.strokeChanged.emit([])

    def set_projection(self, pts):
        """HOOK: recibe la trayectoria futura estimada [(x, y), ...]."""
        self.projection = list(pts)
        self.update()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            if not self._drawing and self.points:
                self.points, self.projection = [], []
            self._drawing = True
            self._add(e.pos())

    def mouseMoveEvent(self, e):
        if self._drawing:
            self._add(e.pos())

    def mouseReleaseEvent(self, e):
        self._drawing = False

    def _add(self, pos):
        self.points.append((pos.x(), pos.y()))
        self.update()
        self.strokeChanged.emit(self.points)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        p.setPen(QPen(QColor("#d4d4d0"), 1.5))
        p.setBrush(QColor(BG))
        p.drawRoundedRect(QRectF(1, 1, w - 2, h - 2), 10, 10)

        # cruz central muy tenue
        p.setPen(QPen(QColor(TRACK), 1))
        p.drawLine(w // 2, 16, w // 2, h - 16)
        p.drawLine(16, h // 2, w - 16, h // 2)

        if not self.points:
            p.setPen(QColor("#a8a8a4"))
            p.setFont(font(26))
            p.drawText(self.rect(), Qt.AlignCenter, "Dibuja una letra aquí")

        self._trazo(p, self.projection, QColor(ACCENT), 3, Qt.DotLine)
        self._trazo(p, self.points, QColor(INK), 4, Qt.SolidLine)

    @staticmethod
    def _trazo(p, pts, color, w, style):
        if len(pts) < 2:
            return
        path = QPainterPath(QPointF(*pts[0]))
        for x, y in pts[1:]:
            path.lineTo(x, y)
        p.setPen(QPen(color, w, style, Qt.RoundCap, Qt.RoundJoin))
        p.setBrush(Qt.NoBrush)
        p.drawPath(path)


# ---------------------------------------------------------------- 3D
class SphereViewFallback(QWidget):
    """Respaldo sin OpenGL: esfera fija dibujada con NumPy + QPainter.

    No rota: el usuario escribe directamente sobre ella con el clic izquierdo.
    El movimiento del ratón recorre la superficie con un mapeo azimutal
    equidistante (ver _pick) y los tramos se unen por arcos de círculo máximo,
    así que el trazo "se envuelve" en la esfera: se comprime desde el centro
    hacia los bordes, se afina y se desvanece, como al escribir sobre una bola.
    El anillo azul marca dónde está el cursor mientras dibujas.
    """

    strokeChanged = pyqtSignal(list)  # lista de (x, y, z)

    YAW, PITCH = 0.0, 0.0  # vista frontal fija
    RIM = 1.2              # radio de cursor (en radios) que equivale a 90 grados

    def __init__(self):
        super().__init__()
        self.setMinimumSize(420, 380)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setCursor(Qt.CrossCursor)
        self.points = []       # trazo del usuario, lista de np.array(3)
        self.projection = []   # proyección futura, lista de np.array(3)
        self._drawing = False
        self._cursor = None
        self._grid = self._build_grid()

    # ---- geometría
    @staticmethod
    def _build_grid(n=64):
        """Meridianos y paralelos cada 15 grados como arreglos (n, 3)."""
        t = np.linspace(0, 2 * np.pi, n)
        lines = []
        for a in np.radians(np.arange(0, 180, 15)):  # meridianos
            lines.append(np.stack([np.cos(t) * np.cos(a), np.sin(t),
                                   np.cos(t) * np.sin(a)], axis=1))
        for lat in np.radians(np.arange(-75, 90, 15)):  # paralelos
            lines.append(np.stack([np.cos(lat) * np.cos(t),
                                   np.full_like(t, np.sin(lat)),
                                   np.cos(lat) * np.sin(t)], axis=1))
        return lines

    def _rot(self):
        cy, sy = np.cos(self.YAW), np.sin(self.YAW)
        cp, sp = np.cos(self.PITCH), np.sin(self.PITCH)
        ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
        return rx @ ry

    def view_rotation(self):
        return self._rot()

    def _geom(self):
        c = QPointF(self.width() / 2, self.height() / 2)
        r = min(self.width(), self.height()) / 2 * 0.86
        return c, r

    def _project(self, pts):
        """(n,3) mundo -> (n,2) pantalla y profundidad z (>= 0 = cara visible)."""
        c, r = self._geom()
        q = np.asarray(pts, dtype=float) @ self._rot().T
        return np.stack([c.x() + r * q[:, 0], c.y() - r * q[:, 1]], axis=1), q[:, 2]

    def _pick(self, pos, clamp=False):
        """Del cursor a un punto de la esfera (coordenadas del mundo).

        Mapeo azimutal equidistante: la distancia del cursor al centro es
        proporcional al ángulo recorrido sobre la superficie (rho -> alpha).
        Así el lápiz "rueda" por la esfera: al alejarte del centro el trazo se
        comprime cada vez más en pantalla (proyección ortográfica) y se curva
        alrededor del borde. Devuelve None si el cursor está fuera de la
        esfera; con clamp=True se usa el borde en su lugar.
        """
        c, r = self._geom()
        x, y = (pos.x() - c.x()) / r, -(pos.y() - c.y()) / r
        rho = float(np.hypot(x, y))
        if rho > 1.0 and not clamp:
            return None
        alpha = min(rho / self.RIM, 1.0) * (np.pi / 2)
        dx, dy = (x / rho, y / rho) if rho > 1e-9 else (0.0, 0.0)
        v = np.array([np.sin(alpha) * dx, np.sin(alpha) * dy, np.cos(alpha)])
        return self._rot().T @ v

    # ---- API pública
    def clear(self):
        self.points, self.projection = [], []
        self.update()
        self.strokeChanged.emit([])

    def set_projection(self, pts):
        """HOOK: trayectoria futura estimada [(x, y, z), ...] sobre la esfera."""
        self.projection = [np.asarray(p, dtype=float) for p in pts]
        self.update()

    # ---- eventos
    def mousePressEvent(self, e):
        if e.button() != Qt.LeftButton:
            return
        if self.points:
            self.points, self.projection = [], []
        if self._pick(e.pos()) is None:
            return
        self._drawing = True
        self._add(e.pos())

    def mouseMoveEvent(self, e):
        if self._drawing:
            self._add(e.pos())

    def mouseReleaseEvent(self, e):
        self._drawing = False
        self._cursor = None
        self.update()

    def _add(self, pos):
        self._cursor = QPointF(pos)
        pt = self._pick(pos, clamp=bool(self.points))
        if pt is None:
            return
        if self.points:
            a = self.points[-1]
            ang = float(np.arccos(np.clip(a @ pt, -1.0, 1.0)))
            n = max(1, int(ang / 0.02))
            if ang > 1e-9:
                # interpolación esférica (slerp): arco de círculo máximo
                for k in range(1, n + 1):
                    t = k / n
                    self.points.append(
                        (np.sin((1 - t) * ang) * a + np.sin(t * ang) * pt) / np.sin(ang))
        else:
            self.points.append(pt)
        self.update()
        self.strokeChanged.emit([tuple(float(v) for v in q) for q in self.points])

    # ---- dibujo
    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        p.setPen(QPen(QColor("#d4d4d0"), 1.5))
        p.setBrush(QColor(BG))
        p.drawRoundedRect(QRectF(1, 1, w - 2, h - 2), 10, 10)

        c, r = self._geom()
        g = QRadialGradient(QPointF(c.x() - 0.3 * r, c.y() - 0.35 * r), 1.5 * r)
        g.setColorAt(0.0, QColor("#ffffff"))
        g.setColorAt(1.0, QColor("#d9e3de"))
        p.setPen(QPen(QColor("#b9c4bf"), 1.5))
        p.setBrush(g)
        p.drawEllipse(c, r, r)

        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(150, 170, 160, 90), 1))
        for line in self._grid:
            self._polilinea(p, line)

        self._trazo(p, self.projection, QColor(ACCENT), 4.0, dashed=True)
        self._trazo(p, self.points, QColor(INK), 5.0)

        if self._drawing and self._cursor is not None:
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(QColor(ACCENT), 2))
            p.drawEllipse(self._cursor, 8, 8)

    def _polilinea(self, p, pts):
        """Dibuja sólo los tramos de la cara visible (z >= 0)."""
        xy, z = self._project(pts)
        path = QPainterPath()
        down = False
        for i in range(len(xy)):
            if z[i] >= 0.0:
                pt = QPointF(xy[i, 0], xy[i, 1])
                if down:
                    path.lineTo(pt)
                else:
                    path.moveTo(pt)
                    down = True
            else:
                down = False
        p.drawPath(path)

    def _trazo(self, p, pts, color, ancho, dashed=False):
        """Trazo sobre la esfera: ancho y opacidad dependen de z, de modo que
        hacia el borde la línea se afina y se desvanece (escorzo)."""
        if len(pts) < 2:
            return
        xy, z = self._project(pts)
        for i in range(len(xy) - 1):
            if z[i] < 0.0 or z[i + 1] < 0.0:
                continue
            if dashed and (i // 3) % 2:
                continue
            zm = 0.5 * (z[i] + z[i + 1])
            col = QColor(color)
            col.setAlpha(int(255 * (0.35 + 0.65 * zm)))
            p.setPen(QPen(col, ancho * (0.30 + 0.70 * zm), Qt.SolidLine,
                          Qt.RoundCap, Qt.RoundJoin))
            p.drawLine(QPointF(xy[i, 0], xy[i, 1]), QPointF(xy[i + 1, 0], xy[i + 1, 1]))


class SphereViewGL(gl.GLViewWidget if HAS_GL else QWidget):
    """Visor 3D interactivo (pyqtgraph.opengl) con una esfera semitransparente.

    - Clic izquierdo sobre la esfera: trazar sobre su superficie.
    - Clic derecho (o clic izquierdo fuera de la esfera): rotar.
    - Rueda: zoom.
    El trazo se guarda como puntos (x, y, z) de la esfera unitaria; los tramos
    entre muestras se unen por arcos de círculo máximo (slerp) para que el
    trazo siga la superficie. La trayectoria futura estimada se dibuja punteada.
    """

    strokeChanged = pyqtSignal(list)  # lista de (x, y, z)
    ALTURA = 1.012                    # el trazo flota apenas sobre la superficie

    def __init__(self):
        super().__init__()
        self.setBackgroundColor(QColor(BG))
        self.setMinimumSize(420, 380)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setCursor(Qt.CrossCursor)

        md = gl.MeshData.sphere(rows=40, cols=80, radius=1.0)
        self.addItem(gl.GLMeshItem(
            meshdata=md, smooth=True, shader="shaded", glOptions="translucent",
            color=(0.88, 0.92, 0.90, 0.55), drawEdges=False))
        self.addItem(gl.GLMeshItem(
            meshdata=gl.MeshData.sphere(rows=12, cols=24, radius=1.003),
            smooth=False, drawFaces=False, drawEdges=True, glOptions="translucent",
            edgeColor=(0.04, 0.49, 0.35, 0.30)))
        self.line = gl.GLLinePlotItem(pos=np.zeros((2, 3)), width=4, antialias=True,
                                      color=(0.1, 0.1, 0.1, 1.0))
        self.proj_line = gl.GLLinePlotItem(pos=np.zeros((2, 3)), width=3,
                                           antialias=True, mode="lines",
                                           color=(0.04, 0.49, 0.35, 1.0))
        self.line.setGLOptions("translucent")
        self.proj_line.setGLOptions("translucent")
        self.addItem(self.line)
        self.addItem(self.proj_line)

        self.points = []   # trazo del usuario, lista de np.array(3) unitarios
        self._line_pts = np.zeros((0, 3))
        self._proj_pts = np.zeros((0, 3))
        self._mode = None  # None | "draw" | "orbit" | "super"
        self._last = QPointF()
        self.reset_view()

    # ---- API pública
    def reset_view(self):
        self.setCameraPosition(distance=4.2, elevation=20, azimuth=30)
        self._refresh()

    def view_rotation(self):
        """Matriz 3x3 que transforma puntos del mundo al marco de la cámara."""
        m = np.array(self.viewMatrix().data(), dtype=float).reshape(4, 4).T
        return m[:3, :3]

    def clear(self):
        self.points = []
        self._line_pts = np.zeros((0, 3))
        self._proj_pts = np.zeros((0, 3))
        self._refresh()
        self.strokeChanged.emit([])

    def set_projection(self, pts):
        """HOOK: trayectoria futura estimada [(x, y, z), ...], dibujada punteada."""
        arr = np.asarray(pts, dtype=float).reshape(-1, 3)
        idx = [i for i in range(len(arr) - 1) if (i // 3) % 2 == 0]
        if not idx:
            self._proj_pts = np.zeros((0, 3))
        else:
            seg = np.empty((2 * len(idx), 3))
            seg[0::2] = arr[idx]
            seg[1::2] = arr[[i + 1 for i in idx]]
            self._proj_pts = seg
        self._refresh()

    # ---- sombreado: el trazo se aclara hacia el borde de la esfera
    def _shade(self, pts, base):
        """Color por vértice. Con f = cos(ángulo entre el punto y la cámara):
        f=1 (de frente) -> color pleno; f->0 (en el borde) -> casi transparente
        y claro; f<0 (cara oculta) -> apenas visible. Así el trazo parece
        realmente escrito sobre la superficie curva."""
        cam = self.cameraPosition()
        c = np.array([cam.x(), cam.y(), cam.z()])
        c /= np.linalg.norm(c)
        f = pts @ c
        w = np.clip(f, 0.0, 1.0) ** 0.7
        light = np.array([0.88, 0.90, 0.89])
        rgb = np.asarray(base)[None, :] * w[:, None] + light[None, :] * (1 - w[:, None])
        alpha = np.where(f > 0, 0.10 + 0.90 * w, 0.05)
        return np.column_stack([rgb, alpha])

    def _refresh(self):
        for item, pts, base in ((self.line, self._line_pts, (0.10, 0.10, 0.10)),
                                (self.proj_line, self._proj_pts, (0.04, 0.49, 0.35))):
            if len(pts) < 2:
                item.setData(pos=np.zeros((2, 3)), color=np.zeros((2, 4)))
            else:
                item.setData(pos=pts * self.ALTURA, color=self._shade(pts, base))

    def orbit(self, azim, elev):
        super().orbit(azim, elev)
        self._refresh()  # el sombreado depende de la dirección de la cámara

    # ---- picking: rayo de la cámara contra la esfera unitaria
    def _pick(self, pos):
        w, h = self.width(), self.height()
        reg = (0, 0, w, h)
        inv, ok = (self.projectionMatrix(reg, reg) * self.viewMatrix()).inverted()
        if not ok:
            return None
        nx, ny = 2 * pos.x() / w - 1, 1 - 2 * pos.y() / h
        a = inv.map(QVector3D(nx, ny, -1))
        b = inv.map(QVector3D(nx, ny, 1))
        o = np.array([a.x(), a.y(), a.z()])
        d = np.array([b.x() - a.x(), b.y() - a.y(), b.z() - a.z()])
        d /= np.linalg.norm(d)
        bq = o @ d
        disc = bq * bq - (o @ o - 1.0)
        if disc < 0:
            return None
        pt = o + (-bq - np.sqrt(disc)) * d  # primera intersección (cara visible)
        return pt / np.linalg.norm(pt)

    # ---- eventos
    def mousePressEvent(self, e):
        self._last = QPointF(e.pos())
        if e.button() == Qt.LeftButton:
            if self._pick(e.pos()) is not None:
                if self.points:
                    self.clear()
                self._mode = "draw"
                self._add(e.pos())
            else:
                self._mode = "orbit"
        elif e.button() == Qt.RightButton:
            self._mode = "orbit"
        else:
            self._mode = "super"
            super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._mode == "draw":
            self._add(e.pos())
        elif self._mode == "orbit":
            d = QPointF(e.pos()) - self._last
            self._last = QPointF(e.pos())
            self.orbit(-d.x() * 0.5, d.y() * 0.5)
        elif self._mode == "super":
            super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if self._mode == "super":
            super().mouseReleaseEvent(e)
        self._mode = None

    def _add(self, pos):
        pt = self._pick(pos)
        if pt is None:
            return
        if self.points:
            a = self.points[-1]
            ang = float(np.arccos(np.clip(a @ pt, -1.0, 1.0)))
            if ang < 1e-6:
                return
            n = max(1, int(ang / 0.02))
            for k in range(1, n + 1):
                t = k / n
                self.points.append(
                    (np.sin((1 - t) * ang) * a + np.sin(t * ang) * pt) / np.sin(ang))
        else:
            self.points.append(pt)
        self._line_pts = np.array(self.points)
        self._refresh()
        self.strokeChanged.emit([tuple(float(v) for v in q) for q in self.points])


SphereView = SphereViewGL if HAS_GL else SphereViewFallback


# ---------------------------------------------------------------- panel
class ProbRow(QWidget):
    """Fila: letra manuscrita · barra · porcentaje."""

    def __init__(self, letra):
        super().__init__()
        self.letra, self.value, self.hot = letra, 0.0, False
        self.setFixedHeight(54)

    def set_state(self, value, hot):
        self.value, self.hot = value, hot
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        if self.hot:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(ACCENT_SOFT))
            p.drawRoundedRect(QRectF(0, 3, w, h - 6), 8, 8)
        dibujar_glifo(p, self.letra, QRectF(10, 6, 34, h - 12),
                      ACCENT if self.hot else INK, 2.6)

        x0, x1 = 62, w - 82
        bar = QRectF(x0, h / 2 - 5, x1 - x0, 10)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#dcdcd8" if self.hot else TRACK))
        p.drawRoundedRect(bar, 5, 5)
        if self.value > 0:
            p.setBrush(QColor(ACCENT if self.hot else "#8a8a86"))
            p.drawRoundedRect(QRectF(bar.x(), bar.y(),
                                     max(10, bar.width() * min(self.value, 1.0)),
                                     bar.height()), 5, 5)

        p.setPen(QColor(ACCENT if self.hot else MUTED))
        p.setFont(font(16, self.hot))
        p.drawText(QRectF(w - 76, 0, 66, h), Qt.AlignVCenter | Qt.AlignRight,
                   f"{self.value * 100:.1f}%")


class ProbabilityPanel(QWidget):
    """Listado ordenado descendentemente de P(carácter | O1:t)."""

    def __init__(self):
        super().__init__()
        self.setFixedWidth(380)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self.top_pct = etiqueta("", 24, semibold=True)
        lay.addWidget(self.top_pct)
        lay.addSpacing(6)
        lay.addSpacing(8)
        lay.addWidget(regla())

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        inner = QWidget()
        inner.setStyleSheet("background: transparent;")
        self.rows_layout = QVBoxLayout(inner)
        self.rows_layout.setContentsMargins(0, 6, 8, 0)
        self.rows_layout.setSpacing(0)
        self.rows = {c: ProbRow(c) for c in ALFABETO}
        for r in self.rows.values():
            self.rows_layout.addWidget(r)
        self.rows_layout.addStretch(1)
        scroll.setWidget(inner)
        lay.addWidget(scroll, 1)
        self.reset()

    def reset(self):
        self.update_probabilities({c: 0.0 for c in ALFABETO})
        self.top_pct.setText("Esperando trazo")

    def update_probabilities(self, probs):
        """probs: dict letra -> probabilidad en [0, 1]."""
        orden = sorted(ALFABETO, key=lambda c: -probs.get(c, 0.0))
        for i, c in enumerate(orden):
            row = self.rows[c]
            row.set_state(probs.get(c, 0.0), i == 0 and probs.get(c, 0.0) > 0)
            self.rows_layout.removeWidget(row)
            self.rows_layout.insertWidget(i, row)
        best = orden[0]
        if probs.get(best, 0.0) > 0:
            self.top_pct.setText(f"Reconociendo: {probs[best] * 100:.1f}%")


# ---------------------------------------------------------------- barra lateral
class NavItem(QWidget):
    clicked = pyqtSignal(int)

    def __init__(self, texto, page):
        super().__init__()
        self.texto, self.page = texto, page
        self.sel = self.hover = False
        self.setFixedHeight(48)
        self.setCursor(Qt.PointingHandCursor)

    def enterEvent(self, _):
        self.hover = True
        self.update()

    def leaveEvent(self, _):
        self.hover = False
        self.update()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton and self.rect().contains(e.pos()):
            self.clicked.emit(self.page)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(0, 2, 0, -2)
        if self.sel or self.hover:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(ACCENT_SOFT if self.sel else SIDE_HOVER))
            p.drawRoundedRect(r, 8, 8)
        p.setPen(QColor(ACCENT if self.sel else INK))
        p.setFont(font(18, self.sel))
        p.drawText(r.adjusted(16, 0, 0, 0), Qt.AlignVCenter, self.texto)


class Sidebar(QWidget):
    def __init__(self, nav):
        super().__init__()
        self.setFixedWidth(250)
        self.setStyleSheet(f"background: {SIDE};")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 30, 18, 24)
        lay.setSpacing(4)

        lay.addWidget(etiqueta("Reconocedor HMM", 22, semibold=True))
        lay.addWidget(etiqueta("Procesos Estocásticos", 15, muted=True))
        lay.addSpacing(34)

        self.items = []
        for texto, page in (("Inicio", PAGE_HOME), ("Modo 2D", PAGE_2D),
                            ("Modo 3D", PAGE_3D), ("Guía de letras", PAGE_GUIA)):
            it = NavItem(texto, page)
            it.clicked.connect(nav)
            self.items.append(it)
            lay.addWidget(it)
        lay.addStretch(1)
        lay.addWidget(etiqueta("Universidad de los Llanos", 14, muted=True))

    def select(self, page):
        for it in self.items:
            it.sel = it.page == page
            it.update()


# ---------------------------------------------------------------- inicio
class Tile(QWidget):
    """Tarjeta grande de modo, con icono lineal."""

    clicked = pyqtSignal()

    def __init__(self, titulo, sub, kind):
        super().__init__()
        self.titulo, self.sub, self.kind, self.hover = titulo, sub, kind, False
        self.setMinimumSize(280, 320)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setFixedHeight(340)
        self.setCursor(Qt.PointingHandCursor)

    def enterEvent(self, _):
        self.hover = True
        self.update()

    def leaveEvent(self, _):
        self.hover = False
        self.update()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton and self.rect().contains(e.pos()):
            self.clicked.emit()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        r = QRectF(1, 1, w - 2, h - 2)
        p.setPen(QPen(QColor(ACCENT if self.hover else "#d4d4d0"), 1.5))
        p.setBrush(QColor(ACCENT_SOFT) if self.hover else QColor(BG))
        p.drawRoundedRect(r, 14, 14)

        col = QColor(ACCENT if self.hover else INK)
        cx, cy = w / 2, h * 0.38
        pen = QPen(col, 3, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        if self.kind == "2d":
            p.drawRoundedRect(QRectF(cx - 60, cy - 50, 120, 100), 8, 8)
            dibujar_glifo(p, "a", QRectF(cx - 40, cy - 36, 80, 72), col, 3)
        else:
            rad = 52
            p.drawEllipse(QPointF(cx, cy), rad, rad)
            p.drawEllipse(QPointF(cx, cy), rad, rad * 0.32)
            p.drawEllipse(QPointF(cx, cy), rad * 0.32, rad)

        p.setPen(col)
        p.setFont(font(28, True))
        p.drawText(QRectF(0, h * 0.64, w, 40), Qt.AlignCenter, self.titulo)
        p.setPen(QColor(MUTED))
        p.setFont(font(17))
        p.drawText(QRectF(0, h * 0.64 + 42, w, 30), Qt.AlignCenter, self.sub)


class HomeView(QWidget):
    def __init__(self, nav):
        super().__init__()
        root = QVBoxLayout(self)
        root.setContentsMargins(64, 56, 64, 48)
        root.setSpacing(0)

        root.addWidget(etiqueta("Reconocedor HMM 2D/3D", 44, semibold=True))
        root.addSpacing(8)
        root.addWidget(etiqueta("Dibuja una letra y el modelo oculto de Markov "
                                "la reconoce mientras la trazas.", 19, muted=True))
        root.addSpacing(40)

        row = QHBoxLayout()
        row.setSpacing(28)
        t2 = Tile("Modo 2D", "Plano cartesiano", "2d")
        t3 = Tile("Modo 3D", "Superficie esférica", "3d")
        t2.clicked.connect(lambda: nav(PAGE_2D))
        t3.clicked.connect(lambda: nav(PAGE_3D))
        row.addWidget(t2)
        row.addWidget(t3)
        root.addLayout(row)
        root.addStretch(1)


# ---------------------------------------------------------------- guía
class GuidePreview(QWidget):
    """Letra grande que se dibuja sola, trazo a trazo, en bucle."""

    def __init__(self):
        super().__init__()
        self.letra, self.u = "a", 0.0
        self.setMinimumSize(380, 380)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(24)
        self.set_letter("a")

    def set_letter(self, letra):
        self.letra, self.u = letra, 0.0
        self.muestras = Glifos.muestras(letra)
        self.update()

    def _tick(self):
        if not self.isVisible():
            return
        self.u += 0.018
        if self.u > len(self.muestras) + 0.9:
            self.u = 0.0
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        p.setPen(QPen(QColor("#d4d4d0"), 1.5))
        p.setBrush(QColor(BG))
        p.drawRoundedRect(QRectF(1, 1, w - 2, h - 2), 10, 10)

        r = QRectF(self.rect()).adjusted(40, 36, -40, -36)
        s = min(r.width() / Glifos.ANCHO, r.height() / Glifos.ALTO)
        ox = r.center().x() - Glifos.ANCHO * s / 2
        oy = r.center().y() - Glifos.ALTO * s / 2

        p.setPen(QPen(QColor(TRACK), 1.5, Qt.DashLine))
        for yy in (0.03, 0.36, 0.8, 1.0):
            y = oy + yy * s
            p.drawLine(QPointF(24, y), QPointF(w - 24, y))

        def pt(q):
            return QPointF(ox + q.x() * s, oy + q.y() * s)

        p.setPen(QPen(QColor("#dcdcd8"), 9, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        p.setBrush(Qt.NoBrush)
        for m in self.muestras:
            path = QPainterPath(pt(m[0]))
            for q in m[1:]:
                path.lineTo(pt(q))
            p.drawPath(path)

        ink = QPen(QColor(INK), 5, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        tip = None
        for k, m in enumerate(self.muestras):
            prog = max(0.0, min(1.0, self.u - k))
            if prog <= 0:
                continue
            n = max(2, int(prog * (len(m) - 1)) + 1)
            path = QPainterPath(pt(m[0]))
            for q in m[1:n]:
                path.lineTo(pt(q))
            p.setPen(ink)
            p.drawPath(path)
            if prog < 1.0:
                tip = pt(m[n - 1])

        p.setFont(font(14, True))
        for k, m in enumerate(self.muestras):
            c = pt(m[0])
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(ACCENT))
            p.drawEllipse(c, 12, 12)
            p.setPen(QColor(BG))
            p.drawText(QRectF(c.x() - 12, c.y() - 12, 24, 24), Qt.AlignCenter,
                       str(k + 1))
        if tip is not None:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(ACCENT))
            p.drawEllipse(tip, 6, 6)


class GuideCell(QWidget):
    clicked = pyqtSignal(str)

    def __init__(self, letra):
        super().__init__()
        self.letra, self.sel, self.hover = letra, False, False
        self.setMinimumSize(78, 98)
        self.setCursor(Qt.PointingHandCursor)

    def enterEvent(self, _):
        self.hover = True
        self.update()

    def leaveEvent(self, _):
        self.hover = False
        self.update()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton and self.rect().contains(e.pos()):
            self.clicked.emit(self.letra)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(2, 2, -2, -2)
        p.setPen(QPen(QColor(ACCENT if self.sel else LINE), 1.5))
        p.setBrush(QColor(ACCENT_SOFT) if (self.sel or self.hover) else QColor(BG))
        p.drawRoundedRect(r, 10, 10)
        dibujar_glifo(p, self.letra, r.adjusted(14, 10, -14, -10),
                      ACCENT if self.sel else INK, 2.8)


class GuideView(QWidget):
    def __init__(self):
        super().__init__()
        root = QVBoxLayout(self)
        root.setContentsMargins(48, 40, 48, 36)
        root.setSpacing(10)

        root.addWidget(etiqueta("Guía de letras", 38, semibold=True))
        root.addWidget(etiqueta("Sigue el punto numerado y traza sin levantar el "
                                "lápiz. Los trazos extra (punto, travesaño, tilde) "
                                "son opcionales.", 17, muted=True))
        root.addSpacing(12)

        body = QHBoxLayout()
        body.setSpacing(32)

        left = QVBoxLayout()
        self.preview = GuidePreview()
        left.addWidget(self.preview, 1)
        self.info = etiqueta("", 20, semibold=True)
        left.addWidget(self.info)
        body.addLayout(left, 1)

        grid_w = QWidget()
        grid = QGridLayout(grid_w)
        grid.setSpacing(10)
        grid.setContentsMargins(0, 0, 0, 0)
        self.cells = {}
        for i, c in enumerate(ALFABETO):
            cell = GuideCell(c)
            cell.clicked.connect(self.select)
            self.cells[c] = cell
            grid.addWidget(cell, i // 5, i % 5)
        grid_w.setMinimumWidth(520)
        body.addWidget(grid_w, 1)
        root.addLayout(body, 1)
        self.select("a")

    def select(self, letra):
        for c, cell in self.cells.items():
            cell.sel = c == letra
            cell.update()
        self.preview.set_letter(letra)
        n = len(Glifos.GLIFOS[letra])
        self.info.setText(f"Letra {letra}  -  " +
                          ("1 trazo" if n == 1 else f"{n} trazos"))


# ---------------------------------------------------------------- reconocedor
class RecognizerView(QWidget):
    """Estructura común: lienzo a la izquierda, probabilidades a la derecha."""

    def __init__(self, titulo, sub, canvas, hint, nav, modo="2d"):
        super().__init__()
        self.canvas = canvas
        self.modo = modo
        self.extractor = FeatureExtractor()
        self.modelos = cargar_modelos_hmm(modo)
        root = QVBoxLayout(self)
        root.setContentsMargins(48, 40, 48, 36)
        root.setSpacing(10)

        head = QHBoxLayout()
        col = QVBoxLayout()
        col.setSpacing(2)
        col.addWidget(etiqueta(titulo, 38, semibold=True))
        col.addWidget(etiqueta(sub, 17, muted=True))
        head.addLayout(col)
        head.addStretch(1)
        self.btn_back = QPushButton("Volver al Menú")
        self.btn_back.clicked.connect(lambda: nav(PAGE_HOME))
        self.btn_reset = QPushButton("Reiniciar captura")
        self.btn_clear = QPushButton("Limpiar lienzo")
        for b in (self.btn_back, self.btn_reset, self.btn_clear):
            b.setCursor(Qt.PointingHandCursor)
            head.addWidget(b)
        root.addLayout(head)
        root.addSpacing(14)

        body = QHBoxLayout()
        body.setSpacing(40)

        left = QVBoxLayout()
        left.setSpacing(10)
        left.addWidget(canvas, 1)
        left.addWidget(etiqueta(hint, 16, muted=True))
        body.addLayout(left, 1)

        self.panel = ProbabilityPanel()
        body.addWidget(self.panel)
        root.addLayout(body, 1)

        self.btn_clear.clicked.connect(self.limpiar)
        self.btn_reset.clicked.connect(self.reiniciar)
        canvas.strokeChanged.connect(self.on_stroke)

    def reiniciar(self):
        """Reinicia la captura: borra el trazo y, en 3D, devuelve la cámara."""
        self.limpiar()
        if hasattr(self.canvas, "reset_view"):
            self.canvas.reset_view()

    def limpiar(self):
        self.canvas.clear()
        self.panel.reset()

    def on_stroke(self, pts):
        """Evalúa Forward en tiempo real y proyecta la trayectoria futura (Viterbi)."""
        if not pts or len(pts) < 2 or not self.modelos:
            self.panel.reset()
            return

        if self.modo == "2d":
            O = self.extractor.simbolos_2d(pts)
        else:
            rot = self.canvas.view_rotation() if hasattr(self.canvas, "view_rotation") else None
            O = self.extractor.simbolos_3d(pts, rot=rot)

        if len(O) == 0:
            return

        probs = posteriores_desde_modelos(self.modelos, O)
        self.panel.update_probabilities(probs)

        mejor = max(probs, key=probs.get)
        futuros = self.modelos[mejor].simbolos_futuros(O)
        if not futuros:
            for c in sorted(probs, key=probs.get, reverse=True)[1:4]:
                cand = self.modelos[c].simbolos_futuros(O)
                if cand:
                    futuros = cand
                    break
        if self.modo == "2d":
            proy = self.extractor.proyectar_2d(pts[-1], futuros)
        else:
            rot = self.canvas.view_rotation() if hasattr(self.canvas, "view_rotation") else None
            proy = self.extractor.proyectar_3d(pts[-1], futuros, rot=rot)
        self.canvas.set_projection(proy)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(TITULO)
        self.resize(1360, 840)
        self.setMinimumSize(1120, 720)

        central = QWidget()
        lay = QHBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.sidebar = Sidebar(self.ir)
        lay.addWidget(self.sidebar)

        self.stack = QStackedWidget()
        lay.addWidget(self.stack, 1)
        self.setCentralWidget(central)

        sphere = SphereView()
        self.home = HomeView(self.ir)
        self.view_2d = RecognizerView(
            "Modo 2D", "Plano cartesiano", DrawingCanvas(),
            "Mantén presionado el clic izquierdo para trazar.", self.ir, modo="2d")
        self.view_3d = RecognizerView(
            "Modo 3D", "Superficie esférica", sphere,
            "Clic izquierdo sobre la esfera: trazar. Clic derecho: rotar. Rueda: zoom.",
            self.ir, modo="3d")
        self.guia = GuideView()
        for v in (self.home, self.view_2d, self.view_3d, self.guia):
            self.stack.addWidget(v)
        self.ir(PAGE_HOME)

    def ir(self, i):
        self.stack.setCurrentIndex(i)
        self.sidebar.select(i)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
