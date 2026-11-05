"""Motor matemático del reconocedor: HMM implementado desde cero (sólo NumPy).

Modelo por letra:  lambda = (A, B, pi)
    A  (N x N)  matriz de transición   a_ij = P(q_{t+1}=j | q_t=i)
    B  (N x K)  matriz de emisión      b_j(k) = P(O_t=k | q_t=j)
    pi (N)      vector inicial         pi_i = P(q_1=i)

Algoritmos (notación de clase):
    forward           alpha_t(i) = P(O_1:t, q_t=i | lambda)   (escalado)
    backward          beta_t(i)  = P(O_{t+1:T} | q_t=i, lambda)
    viterbi           delta_t(i), psi_t(i) -> camino de estados más probable
    baum_welch        re-estimación (EM) de A, B, pi
    simbolos_futuros  proyección de la trayectoria futura (sección 3.5)

Soporta además variantes de trazo por carácter (`submodelos`), ya que una
misma letra (por ejemplo 'd' o 'i') se escribe en el dataset tanto en 1 trazo
como en 2 trazos o en direcciones opuestas.

Este módulo no importa nada de la interfaz: puede probarse solo.
"""
import numpy as np

EPS = 1e-300


class HMMEngine:
    """HMM discreto izquierda-derecha (Bakis) con K símbolos de observación."""

    def __init__(self, A, B, pi, duraciones=None, submodelos=None):
        self.A = np.asarray(A, dtype=float)
        self.B = np.asarray(B, dtype=float)
        self.pi = np.asarray(pi, dtype=float)
        self.N, self.K = self.B.shape
        self.duraciones = (np.asarray(duraciones, dtype=float)
                           if duraciones is not None
                           else 1.0 / np.maximum(1e-3, 1.0 - np.diag(self.A)))
        # Variantes de estilo (cada una es un HMMEngine puro)
        self.submodelos = list(submodelos) if submodelos else []

    def _variantes(self):
        return self.submodelos if self.submodelos else [self]

    def _mejor_variante(self, O):
        variantes = self._variantes()
        if len(variantes) == 1 or len(O) == 0:
            return variantes[0]
        return max(variantes, key=lambda m: m._log_verosimilitud_simple(O))

    # ------------------------------------------------------------ construcción
    @classmethod
    def inicializar(cls, secuencias, N, K):
        """Inicialización izquierda-derecha por segmentación uniforme.

        Cada secuencia se parte en N tramos iguales; el tramo j alimenta el
        estado j. Con esos conteos se arma B, y la duración media por estado
        fija las probabilidades de permanencia de A.
        """
        conteo = np.full((N, K), 0.25)  # suavizado de Laplace
        for O in secuencias:
            T = len(O)
            for t, o in enumerate(O):
                conteo[min(N - 1, t * N // T), o] += 1.0
        B = conteo / conteo.sum(axis=1, keepdims=True)

        L = float(np.mean([len(O) for O in secuencias]))
        a_stay = float(np.clip(1.0 - N / max(1.0, L), 0.2, 0.85))
        A = np.zeros((N, N))
        for i in range(N - 1):
            A[i, i] = a_stay
            resto = 1.0 - a_stay
            if i + 2 < N:
                A[i, i + 1], A[i, i + 2] = 0.85 * resto, 0.15 * resto
            else:
                A[i, i + 1] = resto
        A[N - 1, N - 1] = 1.0
        pi = np.zeros(N)
        pi[0] = 1.0
        dur = np.full(N, max(1.0, L / N))
        return cls(A, B, pi, dur)

    # ------------------------------------------------------------ forward
    def forward(self, O):
        """alpha escalado (T x N) y factores de escala c_t.

        alpha_1(i)     = pi_i b_i(O_1)
        alpha_{t+1}(j) = [ sum_i alpha_t(i) a_ij ] b_j(O_{t+1})

        Se normaliza cada alpha_t (suma 1) dividiendo por c_t = sum_i a_t(i)
        para evitar underflow numérico; así log P(O | lambda) = sum_t log c_t.
        """
        T = len(O)
        alpha = np.zeros((T, self.N))
        c = np.zeros(T)
        a = self.pi * self.B[:, O[0]]
        c[0] = a.sum() + EPS
        alpha[0] = a / c[0]
        for t in range(1, T):
            a = (alpha[t - 1] @ self.A) * self.B[:, O[t]]
            c[t] = a.sum() + EPS
            alpha[t] = a / c[t]
        return alpha, c

    def _log_verosimilitud_simple(self, O):
        if len(O) == 0:
            return 0.0
        _, c = self.forward(O)
        return float(np.sum(np.log(c)))

    def log_verosimilitud(self, O):
        """log P(O_1:t | lambda_c) usando el algoritmo Forward."""
        if len(O) == 0:
            return 0.0
        return max(m._log_verosimilitud_simple(O) for m in self._variantes())

    # ------------------------------------------------------------ backward
    def backward(self, O, c):
        """beta escalado con los mismos factores c_t del forward.

        beta_T(i) = 1
        beta_t(i) = [ sum_j a_ij b_j(O_{t+1}) beta_{t+1}(j) ] / c_{t+1}
        """
        T = len(O)
        beta = np.zeros((T, self.N))
        beta[T - 1] = 1.0
        for t in range(T - 2, -1, -1):
            beta[t] = (self.A @ (self.B[:, O[t + 1]] * beta[t + 1])) / c[t + 1]
        return beta

    # ------------------------------------------------------------ viterbi
    def _viterbi_simple(self, O):
        T = len(O)
        with np.errstate(divide="ignore"):
            logA = np.log(np.maximum(self.A, EPS))
            logB = np.log(np.maximum(self.B, EPS))
            logpi = np.log(np.maximum(self.pi, EPS))
        delta = np.zeros((T, self.N))
        psi = np.zeros((T, self.N), dtype=int)
        delta[0] = logpi + logB[:, O[0]]
        for t in range(1, T):
            cand = delta[t - 1][:, None] + logA          # (i, j)
            psi[t] = np.argmax(cand, axis=0)
            delta[t] = cand[psi[t], np.arange(self.N)] + logB[:, O[t]]
        camino = np.zeros(T, dtype=int)
        camino[-1] = int(np.argmax(delta[-1]))
        for t in range(T - 2, -1, -1):
            camino[t] = psi[t + 1, camino[t + 1]]
        return camino, float(delta[-1].max())

    def viterbi(self, O):
        """Camino de estados ocultos más probable Q* = {q*_1, ..., q*_t}."""
        return self._mejor_variante(O)._viterbi_simple(O)

    # ------------------------------------------------------------ Baum-Welch
    def baum_welch(self, secuencias, iteraciones=10, tol=1e-4, suavizado=0.012):
        """Entrena (A, B, pi) maximizando la verosimilitud (algoritmo EM).

        gamma_t(i) = alpha_t(i) beta_t(i) / sum_j alpha_t(j) beta_t(j)
        xi_t(i,j)  = alpha_t(i) a_ij b_j(O_{t+1}) beta_{t+1}(j) / c_{t+1}
        pi_i   = media de gamma_1(i)
        a_ij   = sum_t xi_t(i,j) / sum_t gamma_t(i)
        b_j(k) = sum_{t:O_t=k} gamma_t(j) / sum_t gamma_t(j)
        """
        previo = -np.inf
        for _ in range(iteraciones):
            pi_num = np.zeros(self.N)
            A_num = np.zeros((self.N, self.N))
            A_den = np.zeros(self.N)
            B_num = np.zeros((self.N, self.K))
            B_den = np.zeros(self.N)
            ll = 0.0
            for O in secuencias:
                T = len(O)
                alpha, c = self.forward(O)
                beta = self.backward(O, c)
                ll += float(np.sum(np.log(c)))
                gamma = alpha * beta
                gamma /= gamma.sum(axis=1, keepdims=True) + EPS
                pi_num += gamma[0]
                if T > 1:
                    xi = (alpha[:-1, :, None] * self.A[None, :, :] *
                          (self.B[:, O[1:]].T * beta[1:] / c[1:, None])[:, None, :])
                    A_num += xi.sum(axis=0)
                    A_den += gamma[:-1].sum(axis=0)
                np.add.at(B_num.T, O, gamma)
                B_den += gamma.sum(axis=0)

            n = max(1, len(secuencias))
            self.pi = (pi_num + 1e-6) / (pi_num.sum() + 1e-6 * self.N)
            vivo = A_den > 1e-9
            self.A[vivo] = A_num[vivo] / A_den[vivo][:, None]
            self.A /= np.maximum(EPS, self.A.sum(axis=1, keepdims=True))
            B = B_num + suavizado
            vivo_b = B_den > 1e-9
            self.B[vivo_b] = B[vivo_b] / B[vivo_b].sum(axis=1, keepdims=True)
            self.duraciones = np.maximum(1.0, B_den / n)
            if abs(ll - previo) < tol * max(1.0, abs(ll)):
                break
            previo = ll
        return ll

    # ------------------------------------------------------------ proyección
    def simbolos_futuros(self, O, maximo=60):
        """Símbolos restantes {O_{t+1}, ..., O_T} de la trayectoria aprendida."""
        if len(O) == 0:
            return []
        m = self._mejor_variante(O)
        camino, _ = m._viterbi_simple(O)
        i = int(camino[-1])
        llevo = 1
        while llevo < len(camino) and camino[-1 - llevo] == i:
            llevo += 1
        futuros = []
        resta = max(0, int(round(m.duraciones[i])) - llevo)
        futuros += [int(np.argmax(m.B[i]))] * resta
        for j in range(i + 1, m.N):
            futuros += [int(np.argmax(m.B[j]))] * max(1, int(round(m.duraciones[j])))
        return futuros[:maximo]

    # ------------------------------------------------------------ persistencia
    def a_dict(self):
        d = {
            "A": np.round(self.A, 6).tolist(),
            "B": np.round(self.B, 6).tolist(),
            "pi": np.round(self.pi, 6).tolist(),
            "duraciones": np.round(self.duraciones, 3).tolist(),
        }
        if self.submodelos and len(self.submodelos) > 1:
            d["variantes"] = [
                {
                    "A": np.round(m.A, 6).tolist(),
                    "B": np.round(m.B, 6).tolist(),
                    "pi": np.round(m.pi, 6).tolist(),
                    "duraciones": np.round(m.duraciones, 3).tolist(),
                }
                for m in self.submodelos
            ]
        return d

    @classmethod
    def desde_dict(cls, d):
        subs = [cls(v["A"], v["B"], v["pi"], v.get("duraciones")) for v in d.get("variantes", [])]
        return cls(d["A"], d["B"], d["pi"], d.get("duraciones"), submodelos=subs)


def posteriores_desde_modelos(modelos, O, temperatura=None):
    """Evalúa P(c | O_1:t) para todas las letras usando Forward y Bayes."""
    if len(O) == 0 or not modelos:
        return {c: 0.0 for c in modelos}
    letras = list(modelos.keys())
    lls = np.array([modelos[c].log_verosimilitud(O) for c in letras], dtype=float)
    tau = temperatura if temperatura is not None else max(1.5, 0.35 * len(O))
    z = (lls - lls.max()) / tau
    w = np.exp(np.clip(z, -60.0, 0.0))
    w /= w.sum() + EPS
    return {c: float(p) for c, p in zip(letras, w)}
