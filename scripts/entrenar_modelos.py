"""Entrenamiento offline de los 27 modelos HMM (uno por letra, para 2D y 3D).

Lee data/dataset_compartido.json, extrae las secuencias de observaciones con
FeatureExtractor (código de cadena en 2D y variaciones esféricas d_theta, d_phi
en 3D), entrena un modelo HMM nativo por cada letra mediante Baum-Welch
(HMMEngine) y exporta los parámetros aprendidos a:
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

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "src"))
from FeatureExtractor import FeatureExtractor  # noqa: E402
from HMMEngine import HMMEngine  # noqa: E402

RUTA_DATASET = os.path.join(RAIZ, "data", "dataset_compartido.json")
DIR_MODELOS = os.path.join(RAIZ, "models")


def cargar_dataset(ruta=RUTA_DATASET):
    """Carga el dataset JSON compartido."""
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def preparar_secuencias(dataset, modo="2d", fx=None):
    """Extrae características y devuelve {letra: [secuencia_O, ...]}."""
    fx = fx or FeatureExtractor()
    secuencias = defaultdict(list)

    # Formato 1: diccionario {letra: [muestra, ...]} (glifo-dataset)
    if isinstance(dataset, dict):
        items = [(letra, muestra) for letra, muestras in dataset.items() for muestra in muestras]
        for letra, muestra in items:
            if modo == "2d":
                O = fx.simbolos_dataset_2d(muestra)
            else:
                pts_3d = fx.muestra_2d_a_esfera(muestra)
                O = fx.simbolos_3d(pts_3d)
            if len(O) >= 3:
                secuencias[letra].append([int(k) for k in O])
    # Formato 2: lista de objetos [{"letra": "a", "modo": "2d", "puntos": [...]}, ...]
    elif isinstance(dataset, list):
        for item in dataset:
            if item.get("modo", modo) != modo:
                continue
            pts = item["puntos"]
            O = fx.simbolos_dataset_2d(pts) if modo == "2d" else fx.simbolos_3d(pts)
            if len(O) >= 3:
                secuencias[item["letra"]].append([int(k) for k in O])

    return dict(secuencias)


def entrenar_modelos(secuencias, N=10, K=FeatureExtractor.K, iteraciones=10):
    """Entrena un HMM independiente lambda_c = (A_c, B_c, pi_c) por cada letra."""
    modelos = {}
    for letra, seqs in sorted(secuencias.items()):
        hmm = HMMEngine.inicializar(seqs, N=N, K=K)
        hmm.baum_welch(seqs, iteraciones=iteraciones)
        modelos[letra] = hmm
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

        if args.evaluar:
            ent, pru = {}, {}
            for c, s in secuencias.items():
                corte = max(1, int(0.8 * len(s)))
                ent[c], pru[c] = s[:corte], s[corte:]
            m_eval = entrenar_modelos(ent, N=args.estados, iteraciones=args.iteraciones)
            top1, top3 = evaluar(m_eval, pru)
            print(f"[{modo.upper()}] Exactitud prueba -> Top-1: {100*top1:.1f}% | Top-3: {100*top3:.1f}%")

        print(f"[{modo.upper()}] Entrenando modelos finales con Baum-Welch (N={args.estados})...")
        modelos = entrenar_modelos(secuencias, N=args.estados, iteraciones=args.iteraciones)
        ruta = guardar_modelos(modelos, modo, N=args.estados)
        print(f"[{modo.upper()}] Guardado en: {ruta}")


if __name__ == "__main__":
    main()
