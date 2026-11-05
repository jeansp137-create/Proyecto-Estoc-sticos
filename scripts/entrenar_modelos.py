"""Entrenamiento offline de los 27 modelos HMM (uno por letra, para 2D y 3D).

Lee data/dataset_compartido.json, extrae las secuencias de observaciones con
FeatureExtractor (código de cadena extendido en 2D y variaciones esféricas
d_theta, d_phi en 3D), separa automáticamente las variantes de estilo de cada
letra (1 trazo vs. varios trazos, dirección arriba/abajo y estilo de la Guía),
entrena los HMM con Baum-Welch (HMMEngine) y exporta los parámetros a:
    models/modelos_2d.json
    models/modelos_3d.json

Uso:
    python scripts/entrenar_modelos.py [--modo 2d|3d|ambos] [--estados 10]
                                       [--iteraciones 10] [--evaluar]
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "src"))
from FeatureExtractor import FeatureExtractor  # noqa: E402
from HMMEngine import HMMEngine  # noqa: E402
import Glifos  # noqa: E402

RUTA_DATASET = os.path.join(RAIZ, "data", "dataset_compartido.json")
DIR_MODELOS = os.path.join(RAIZ, "models")


def cargar_dataset(ruta=RUTA_DATASET):
    """Carga el dataset JSON compartido."""
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def _secuencias_guia(letra, modo="2d", fx=None, n_muestras=24):
    """Genera secuencias de observación a partir del glifo de la Guía de letras."""
    fx = fx or FeatureExtractor()
    rng = np.random.default_rng(100 + ord(letra[0]))
    trazos0 = [[(q.x() * 300.0, q.y() * 300.0) for q in tr] for tr in Glifos.muestras(letra, 35)]
    seqs = []
    for _ in range(n_muestras):
        sx, sy = rng.uniform(0.78, 1.30, 2)
        muestra = []
        for tr in trazos0:
            P = np.asarray(tr, dtype=float)
            if len(P) >= 2 and np.hypot(*np.diff(P, axis=0).T).sum() < 8.0:
                P = P[:1]  # punto de i / j
            P = (P * [sx, sy] + rng.normal(0.0, 2.0, P.shape)).tolist()
            muestra.append(P)
        if modo == "2d":
            O = fx.simbolos_2d(muestra)
        else:
            pts_3d = fx.muestra_2d_a_esfera(muestra)
            O = fx.simbolos_3d(pts_3d)
        if len(O) >= 2:
            seqs.append([int(k) for k in O])
    return seqs


def _agrupar_estilos(letra, s_list, modo="2d", fx=None):
    """Divide las secuencias de una letra en grupos homogéneos de trazo."""
    grupos_por_tipo = defaultdict(list)
    for q in s_list:
        grupos_por_tipo[(16 in q, 17 in q)].append(q)

    clusters = []
    for _, sub in grupos_por_tipo.items():
        if len(sub) < 8:
            continue
        if len(sub) < 30:
            clusters.append(sub)
            continue
        vecs = []
        for q in sub:
            ang_only = [x for x in q if x < 16] or [0]
            ang = 2 * np.pi * np.asarray(ang_only, dtype=float) / 16.0
            parts_c = np.array_split(np.cos(ang), 4)
            parts_s = np.array_split(np.sin(ang), 4)
            v = ([float(np.mean(p)) if len(p) else 0.0 for p in parts_c] +
                 [float(np.mean(p)) if len(p) else 0.0 for p in parts_s])
            vecs.append(v)
        X = np.asarray(vecs, dtype=float)
        idx0 = 0
        idx1 = int(np.argmax(np.linalg.norm(X - X[idx0], axis=1)))
        C = X[[idx0, idx1]].copy()
        lab = np.zeros(len(sub), dtype=int)
        for _ in range(8):
            d = np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2)
            lab = np.argmin(d, axis=1)
            for k in range(2):
                if np.any(lab == k):
                    C[k] = X[lab == k].mean(axis=0)
        for k in range(2):
            cl = [sub[i] for i in range(len(sub)) if lab[i] == k]
            if len(cl) >= 8:
                clusters.append(cl)

    guia = _secuencias_guia(letra, modo=modo, fx=fx)
    if guia:
        clusters.append(guia)
    if not clusters and s_list:
        clusters = [s_list]
    return clusters


def preparar_secuencias(dataset, modo="2d", fx=None):
    """Extrae características y devuelve {letra: [secuencia_O, ...]}."""
    fx = fx or FeatureExtractor()
    secuencias = defaultdict(list)

    if isinstance(dataset, dict):
        for letra, muestras in dataset.items():
            for muestra in muestras:
                if modo == "2d":
                    O = fx.simbolos_dataset_2d(muestra)
                else:
                    pts_3d = fx.muestra_2d_a_esfera(muestra)
                    O = fx.simbolos_3d(pts_3d)
                if len(O) >= 2:
                    secuencias[letra].append([int(k) for k in O])
    elif isinstance(dataset, list):
        for item in dataset:
            if item.get("modo", modo) != modo:
                continue
            pts = item["puntos"]
            O = fx.simbolos_dataset_2d(pts) if modo == "2d" else fx.simbolos_3d(pts)
            if len(O) >= 2:
                secuencias[item["letra"]].append([int(k) for k in O])

    return dict(secuencias)


def entrenar_modelos(secuencias, N=10, K=FeatureExtractor.K, iteraciones=10, modo="2d"):
    """Entrena un HMM (con sus variantes de trazo) por cada letra."""
    fx = FeatureExtractor()
    modelos = {}
    for letra, seqs in sorted(secuencias.items()):
        clusters = _agrupar_estilos(letra, seqs, modo=modo, fx=fx)
        variantes = []
        for cl in clusters:
            l_med = float(np.median([len(q) for q in cl]))
            n_estados = int(np.clip(round(l_med * 0.6), 4, N))
            h = HMMEngine.inicializar(cl, N=n_estados, K=K)
            h.baum_welch(cl, iteraciones=iteraciones, suavizado=0.012)
            variantes.append(h)
        principal = variantes[0]
        modelos[letra] = HMMEngine(
            principal.A, principal.B, principal.pi, principal.duraciones, submodelos=variantes
        )
    return modelos


def evaluar(modelos, secuencias):
    """Devuelve (exactitud_top1, exactitud_top3) sobre las secuencias dadas."""
    ok1 = ok3 = total = 0
    for letra, seqs in secuencias.items():
        for O in seqs:
            ranking = sorted(modelos.keys(), key=lambda c: -modelos[c].log_verosimilitud(O))
            if ranking[0] == letra:
                ok1 += 1
            if letra in ranking[:3]:
                ok3 += 1
            total += 1
    return ok1 / max(1, total), ok3 / max(1, total)


def guardar_modelos(modelos, modo, directorio=DIR_MODELOS, N=10, K=FeatureExtractor.K):
    """Guarda los modelos entrenados en models/modelos_<modo>.json."""
    os.makedirs(directorio, exist_ok=True)
    ruta = os.path.join(directorio, f"modelos_{modo}.json")
    datos = {
        "modo": modo,
        "estados": N,
        "simbolos": K,
        "letras": {c: m.a_dict() for c, m in sorted(modelos.items())},
    }
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(datos, f, ensure_ascii=False, separators=(",", ":"))
    return ruta


def cargar_modelos(modo, directorio=DIR_MODELOS):
    """Carga models/modelos_<modo>.json y devuelve {letra: HMMEngine}."""
    ruta = os.path.join(directorio, f"modelos_{modo}.json")
    if not os.path.exists(ruta):
        return {}
    with open(ruta, encoding="utf-8") as f:
        datos = json.load(f)
    return {c: HMMEngine.desde_dict(d) for c, d in datos.get("letras", {}).items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modo", default="ambos", choices=["2d", "3d", "ambos"])
    ap.add_argument("--estados", type=int, default=10)
    ap.add_argument("--iteraciones", type=int, default=10)
    ap.add_argument("--evaluar", action="store_true")
    args = ap.parse_args()

    dataset = cargar_dataset()
    fx = FeatureExtractor()
    modos = ["2d", "3d"] if args.modo == "ambos" else [args.modo]

    for modo in modos:
        print(f"[{modo.upper()}] Extrayendo secuencias del dataset...")
        secuencias = preparar_secuencias(dataset, modo=modo, fx=fx)
        total_muestras = sum(len(v) for v in secuencias.values())
        print(f"[{modo.upper()}] {len(secuencias)} letras, {total_muestras} secuencias válidas.")

        print(f"[{modo.upper()}] Entrenando modelos con Baum-Welch...")
        modelos = entrenar_modelos(
            secuencias, N=args.estados, iteraciones=args.iteraciones, modo=modo
        )
        if args.evaluar:
            top1, top3 = evaluar(modelos, secuencias)
            print(f"[{modo.upper()}] Exactitud -> Top-1: {100*top1:.1f}% | Top-3: {100*top3:.1f}%")
        ruta = guardar_modelos(modelos, modo, N=args.estados)
        print(f"[{modo.upper()}] Guardado en: {ruta}")


if __name__ == "__main__":
    main()
