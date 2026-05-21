"""Librería estándar de DeepLang para Machine Learning.

Contenido:
    1. Activaciones y utilidades matemáticas
    2. Normalización de datos
    3. Métricas de evaluación
    4. Regresión lineal y logística
    5. Perceptrón simple y multicapa (MLP)
    6. K-Means clustering
"""


class DeepLangMLLib:
    """Funciones de Machine Learning para integrarse como builtins."""

    # ─── Estado global para generador pseudoaleatorio ──────────
    _lcg_state = 42

    # ─── Constantes ───────────────────────────────────────────
    PI = 3.141592653589793

    # ─── Activaciones y utilidades ────────────────────────────

    def sig(self, x):
        """Sigmoide: 1 / (1 + e^-x)"""
        x = float(x)
        if x > 500.0:
            return 1.0
        if x < -500.0:
            return 0.0
        return 1.0 / (1.0 + self._exp(-x))

    def sig_deriv(self, s):
        """Derivada de sigmoide: s*(1-s)"""
        s = float(s)
        return s * (1.0 - s)

    def relu(self, x):
        """ReLU: max(0, x)"""
        x = float(x)
        return x if x > 0.0 else 0.0

    def relu_deriv(self, x):
        """Derivada de ReLU"""
        x = float(x)
        return 1.0 if x > 0.0 else 0.0

    def tanh_dl(self, x):
        """Tanh: (e^x - e^-x) / (e^x + e^-x)"""
        x = float(x)
        ep = self._exp(x)
        en = self._exp(-x)
        return (ep - en) / (ep + en)

    def tanh_deriv(self, t):
        """Derivada de tanh: 1 - t^2"""
        t = float(t)
        return 1.0 - t * t

    def _exp(self, x):
        """Exponencial usando serie de Taylor"""
        x = float(x)
        term = 1.0
        result = 1.0
        for n in range(1, 80):
            term *= x / n
            result += term
            if abs(term) < 1e-15:
                break
        return result

    # ─── Normalización ────────────────────────────────────────

    def media(self, datos, n):
        """Media de n elementos"""
        n = int(n)
        if n <= 0 or not isinstance(datos, list):
            raise RuntimeError("media() espera un arreglo no vacío")
        suma = 0.0
        for i in range(n):
            suma += float(datos[i])
        return suma / n

    def desviacion(self, datos, n):
        """Desviación estándar de n elementos"""
        n = int(n)
        if n <= 0 or not isinstance(datos, list):
            raise RuntimeError("desviacion() espera un arreglo no vacío")
        m = self.media(datos, n)
        suma = 0.0
        for i in range(n):
            d = float(datos[i]) - m
            suma += d * d
        std = suma / n
        if std == 0:
            return 0.0
        # Raíz cuadrada de Newton
        guess = std if std >= 1 else 1.0
        for _ in range(40):
            guess = 0.5 * (guess + std / guess)
        return guess

    def normalizar_minmax(self, datos, n):
        """Normalización min-max al rango [0, 1]"""
        n = int(n)
        if n <= 0 or not isinstance(datos, list):
            raise RuntimeError("normalizar_minmax() espera un arreglo no vacío")
        mn = datos[0]
        mx = datos[0]
        for i in range(1, n):
            if float(datos[i]) < float(mn):
                mn = datos[i]
            if float(datos[i]) > float(mx):
                mx = datos[i]
        rng = float(mx) - float(mn)
        norm = []
        for i in range(n):
            if rng < 1e-10:
                norm.append(0.0)
            else:
                norm.append((float(datos[i]) - float(mn)) / rng)
        return norm

    def normalizar_zscore(self, datos, n):
        """Normalización Z-score: (x - media) / std"""
        n = int(n)
        if n <= 0 or not isinstance(datos, list):
            raise RuntimeError("normalizar_zscore() espera un arreglo no vacío")
        m = self.media(datos, n)
        std = self.desviacion(datos, n)
        norm = []
        for i in range(n):
            if std < 1e-10:
                norm.append(0.0)
            else:
                norm.append((float(datos[i]) - m) / std)
        return norm

    # ─── Métricas ─────────────────────────────────────────────

    def mse(self, y_real, y_pred, n):
        """Error cuadrático medio"""
        n = int(n)
        if not isinstance(y_real, list) or not isinstance(y_pred, list):
            raise RuntimeError("mse() espera dos arreglos")
        suma = 0.0
        for i in range(n):
            d = float(y_real[i]) - float(y_pred[i])
            suma += d * d
        return suma / n

    def mae(self, y_real, y_pred, n):
        """Error absoluto medio"""
        n = int(n)
        if not isinstance(y_real, list) or not isinstance(y_pred, list):
            raise RuntimeError("mae() espera dos arreglos")
        suma = 0.0
        for i in range(n):
            d = float(y_real[i]) - float(y_pred[i])
            suma += abs(d)
        return suma / n

    def r_cuadrado(self, y_real, y_pred, n):
        """Coeficiente de determinación R²"""
        n = int(n)
        if not isinstance(y_real, list) or not isinstance(y_pred, list):
            raise RuntimeError("r_cuadrado() espera dos arreglos")
        m = self.media(y_real, n)
        ss_res = 0.0
        ss_tot = 0.0
        for i in range(n):
            d1 = float(y_real[i]) - float(y_pred[i])
            d2 = float(y_real[i]) - m
            ss_res += d1 * d1
            ss_tot += d2 * d2
        if ss_tot < 1e-10:
            return 1.0
        return 1.0 - ss_res / ss_tot

    def accuracy(self, y_real, y_pred, n):
        """Exactitud para clasificación binaria (threshold 0.5)"""
        n = int(n)
        if not isinstance(y_real, list) or not isinstance(y_pred, list):
            raise RuntimeError("accuracy() espera dos arreglos")
        correctos = 0.0
        for i in range(n):
            pred = 1 if float(y_pred[i]) >= 0.5 else 0
            if pred == int(y_real[i]):
                correctos += 1.0
        return correctos / n

    # ─── Regresión lineal (una variable) ──────────────────────

    def regresion_lineal(self, X, Y, n, lr, epochs):
        """
        Regresión lineal y = w*x + b usando gradiente descendente.
        Retorna [w, b]
        """
        n = int(n)
        lr = float(lr)
        epochs = int(epochs)
        if not isinstance(X, list) or not isinstance(Y, list):
            raise RuntimeError("regresion_lineal() espera dos arreglos")

        w = 0.0
        b = 0.0
        for ep in range(epochs):
            grad_w = 0.0
            grad_b = 0.0
            for i in range(n):
                pred = w * float(X[i]) + b
                error = pred - float(Y[i])
                grad_w += error * float(X[i])
                grad_b += error
            grad_w /= n
            grad_b /= n
            w -= lr * grad_w
            b -= lr * grad_b
        return [w, b]

    def rl_predecir(self, params, X, n):
        """Predicción con parámetros de regresión lineal [w, b]"""
        n = int(n)
        if not isinstance(params, list) or len(params) < 2:
            raise RuntimeError("rl_predecir() espera params=[w, b]")
        if not isinstance(X, list):
            raise RuntimeError("rl_predecir() espera arreglo X")
        w = float(params[0])
        b = float(params[1])
        preds = []
        for i in range(n):
            preds.append(w * float(X[i]) + b)
        return preds

    # ─── Regresión lineal múltiple ────────────────────────────

    def regresion_lineal_multi(self, X_plano, Y, n, m, lr, epochs):
        """
        Regresión lineal múltiple: y = w0 + w1*x1 + ... + wm*xm
        X_plano: matriz n x m (elementos en orden fila-mayor)
        Y: arreglo n
        Retorna W[0..m] donde W[0]=bias
        """
        n = int(n)
        m = int(m)
        lr = float(lr)
        epochs = int(epochs)

        if not isinstance(X_plano, list) or not isinstance(Y, list):
            raise RuntimeError("regresion_lineal_multi() espera dos arreglos")

        W = [0.0] * (m + 1)

        for ep in range(epochs):
            grads = [0.0] * (m + 1)
            for i in range(n):
                pred = W[0]
                for j in range(m):
                    pred += W[j + 1] * float(X_plano[i * m + j])
                err = pred - float(Y[i])
                grads[0] += err
                for j in range(m):
                    grads[j + 1] += err * float(X_plano[i * m + j])

            for j in range(m + 1):
                W[j] -= lr * grads[j] / n

        return W

    def rl_multi_predecir(self, W, X_plano, n, m):
        """Predicción regresión lineal múltiple"""
        n = int(n)
        m = int(m)
        if not isinstance(W, list) or len(W) < m + 1:
            raise RuntimeError("rl_multi_predecir() espera W con m+1 elementos")
        if not isinstance(X_plano, list):
            raise RuntimeError("rl_multi_predecir() espera arreglo X_plano")

        preds = []
        for i in range(n):
            pred = float(W[0])
            for j in range(m):
                pred += float(W[j + 1]) * float(X_plano[i * m + j])
            preds.append(pred)
        return preds

    # ─── Regresión logística ──────────────────────────────────

    def regresion_logistica(self, X, Y, n, lr, epochs):
        """
        Regresión logística P(y=1|x) = sigmoide(w*x + b)
        Retorna [w, b]
        """
        n = int(n)
        lr = float(lr)
        epochs = int(epochs)
        if not isinstance(X, list) or not isinstance(Y, list):
            raise RuntimeError("regresion_logistica() espera dos arreglos")

        w = 0.0
        b = 0.0
        for ep in range(epochs):
            grad_w = 0.0
            grad_b = 0.0
            for i in range(n):
                pred = self.sig(w * float(X[i]) + b)
                error = pred - float(Y[i])
                grad_w += error * float(X[i])
                grad_b += error
            w -= lr * grad_w / n
            b -= lr * grad_b / n
        return [w, b]

    def log_predecir(self, params, X, n):
        """Predicción regresión logística"""
        n = int(n)
        if not isinstance(params, list) or len(params) < 2:
            raise RuntimeError("log_predecir() espera params=[w, b]")
        if not isinstance(X, list):
            raise RuntimeError("log_predecir() espera arreglo X")
        w = float(params[0])
        b = float(params[1])
        preds = []
        for i in range(n):
            preds.append(self.sig(w * float(X[i]) + b))
        return preds

    # ─── Regresión logística múltiple ─────────────────────────

    def regresion_logistica_multi(self, X_plano, Y, n, m, lr, epochs):
        """
        Regresión logística múltiple: P(y=1) = sigmoide(w0 + sum(wi*xi))
        X_plano: matriz n x m
        Y: arreglo n con valores {0, 1}
        Retorna W[0..m]
        """
        n = int(n)
        m = int(m)
        lr = float(lr)
        epochs = int(epochs)

        if not isinstance(X_plano, list) or not isinstance(Y, list):
            raise RuntimeError("regresion_logistica_multi() espera dos arreglos")

        W = [0.0] * (m + 1)

        for ep in range(epochs):
            grads = [0.0] * (m + 1)
            for i in range(n):
                z = float(W[0])
                for j in range(m):
                    z += float(W[j + 1]) * float(X_plano[i * m + j])
                pred = self.sig(z)
                err = pred - float(Y[i])
                grads[0] += err
                for j in range(m):
                    grads[j + 1] += err * float(X_plano[i * m + j])

            for j in range(m + 1):
                W[j] -= lr * grads[j] / n

        return W

    def log_multi_predecir(self, W, X_plano, n, m):
        """Predicción regresión logística múltiple"""
        n = int(n)
        m = int(m)
        if not isinstance(W, list) or len(W) < m + 1:
            raise RuntimeError("log_multi_predecir() espera W con m+1 elementos")
        if not isinstance(X_plano, list):
            raise RuntimeError("log_multi_predecir() espera arreglo X_plano")

        preds = []
        for i in range(n):
            z = float(W[0])
            for j in range(m):
                z += float(W[j + 1]) * float(X_plano[i * m + j])
            preds.append(self.sig(z))
        return preds

    # ─── Perceptrón simple ────────────────────────────────────

    def perceptron(self, X_plano, Y, n, m, lr, epochs):
        """
        Perceptrón simple (función de activación escalón).
        X_plano: n x m, Y: {0, 1}
        Retorna W[0..m] (W[0]=bias)
        """
        n = int(n)
        m = int(m)
        lr = float(lr)
        epochs = int(epochs)

        if not isinstance(X_plano, list) or not isinstance(Y, list):
            raise RuntimeError("perceptron() espera dos arreglos")

        W = [0.0] * (m + 1)

        for ep in range(epochs):
            for i in range(n):
                z = float(W[0])
                for j in range(m):
                    z += float(W[j + 1]) * float(X_plano[i * m + j])
                pred = 1 if z >= 0.0 else 0
                err = float(Y[i]) - pred
                W[0] += lr * err
                for j in range(m):
                    W[j + 1] += lr * err * float(X_plano[i * m + j])

        return W

    def perceptron_predecir(self, W, X_plano, n, m):
        """Predicción perceptrón"""
        n = int(n)
        m = int(m)
        if not isinstance(W, list) or len(W) < m + 1:
            raise RuntimeError("perceptron_predecir() espera W con m+1 elementos")
        if not isinstance(X_plano, list):
            raise RuntimeError("perceptron_predecir() espera arreglo X_plano")

        preds = []
        for i in range(n):
            z = float(W[0])
            for j in range(m):
                z += float(W[j + 1]) * float(X_plano[i * m + j])
            pred = 1 if z >= 0.0 else 0
            preds.append(pred)
        return preds

    # ─── Generador pseudoaleatorio ────────────────────────────

    def _rand_next(self):
        """LCG: Linear Congruential Generator"""
        self._lcg_state = (self._lcg_state * 1664525 + 1013904223) % 2147483647
        return self._lcg_state

    def _rand_float(self):
        """Retorna float en (-1, 1)"""
        v = self._rand_next()
        return (v % 10000) / 10000.0 * 2.0 - 1.0

    def ml_semilla(self, s):
        """Establece semilla del generador aleatorio"""
        self._lcg_state = int(s)
        return 0

    # ─── MLP (Perceptrón Multicapa) ───────────────────────────

    def mlp_init(self, n_in, n_hidden, n_out):
        """
        Inicializa una red MLP 3 capas (entrada->oculta->salida).
        Retorna arreglo plano con layout:
        [n_in, n_hidden, n_out, W1[...], b1[...], W2[...], b2[...]]
        """
        n_in = int(n_in)
        n_hidden = int(n_hidden)
        n_out = int(n_out)

        red = [float(n_in), float(n_hidden), float(n_out)]

        # W1: n_in x n_hidden
        for _ in range(n_in * n_hidden):
            red.append(self._rand_float() * 0.5)

        # b1: n_hidden
        for _ in range(n_hidden):
            red.append(0.0)

        # W2: n_hidden x n_out
        for _ in range(n_hidden * n_out):
            red.append(self._rand_float() * 0.5)

        # b2: n_out
        for _ in range(n_out):
            red.append(0.0)

        return red

    def mlp_n_in(self, red):
        """Dimensión de entrada"""
        return int(red[0])

    def mlp_n_hidden(self, red):
        """Dimensión oculta"""
        return int(red[1])

    def mlp_n_out(self, red):
        """Dimensión de salida"""
        return int(red[2])

    def mlp_forward(self, red, x, n_in):
        """
        Forward pass: retorna [h_0..h_hidden, o_0..o_out]
        donde h = activación oculta, o = activación salida
        """
        n_in = int(n_in)
        n_hidden = int(red[1])
        n_out = int(red[2])
        off_W1 = 3
        off_b1 = 3 + n_in * n_hidden
        off_W2 = off_b1 + n_hidden
        off_b2 = off_W2 + n_hidden * n_out

        if not isinstance(x, list) or len(x) < n_in:
            raise RuntimeError("mlp_forward() espera arreglo x de tamaño n_in")

        # Capa oculta
        h = []
        for j in range(n_hidden):
            z = float(red[off_b1 + j])
            for i in range(n_in):
                z += float(red[off_W1 + i * n_hidden + j]) * float(x[i])
            h.append(self.sig(z))

        # Capa salida
        o = []
        for j in range(n_out):
            z = float(red[off_b2 + j])
            for i in range(n_hidden):
                z += float(red[off_W2 + i * n_out + j]) * float(h[i])
            o.append(self.sig(z))

        # Concatenar h + o
        salida = h + o
        return salida

    def mlp_entrenar(self, red, X_plano, Y_plano, n, lr, epochs):
        """
        Entrenamiento MLP con backpropagation.
        X_plano: n x n_in, Y_plano: n x n_out
        Retorna red actualizada
        """
        n = int(n)
        lr = float(lr)
        epochs = int(epochs)
        n_in = int(red[0])
        n_hidden = int(red[1])
        n_out = int(red[2])
        off_W1 = 3
        off_b1 = 3 + n_in * n_hidden
        off_W2 = off_b1 + n_hidden
        off_b2 = off_W2 + n_hidden * n_out

        if not isinstance(X_plano, list) or not isinstance(Y_plano, list):
            raise RuntimeError("mlp_entrenar() espera dos arreglos")

        for ep in range(epochs):
            for i in range(n):
                # Extraer ejemplo
                x = []
                for j in range(n_in):
                    x.append(float(X_plano[i * n_in + j]))
                y = []
                for j in range(n_out):
                    y.append(float(Y_plano[i * n_out + j]))

                # Forward
                act = self.mlp_forward(red, x, n_in)
                h = act[:n_hidden]
                o = act[n_hidden:]

                # Delta salida: (o - y) * o'
                delta_o = []
                for j in range(n_out):
                    delta = (float(o[j]) - float(y[j])) * self.sig_deriv(float(o[j]))
                    delta_o.append(delta)

                # Delta oculta: sum(delta_o_k * W2_jk) * h'
                delta_h = []
                for j in range(n_hidden):
                    suma = 0.0
                    for k in range(n_out):
                        suma += float(delta_o[k]) * float(red[off_W2 + j * n_out + k])
                    delta = suma * self.sig_deriv(float(h[j]))
                    delta_h.append(delta)

                # Actualizar W2 y b2
                for j in range(n_hidden):
                    for k in range(n_out):
                        idx = off_W2 + j * n_out + k
                        red[idx] -= lr * float(delta_o[k]) * float(h[j])
                for k in range(n_out):
                    red[off_b2 + k] -= lr * float(delta_o[k])

                # Actualizar W1 y b1
                for j in range(n_hidden):
                    for ii in range(n_in):
                        idx = off_W1 + ii * n_hidden + j
                        red[idx] -= lr * float(delta_h[j]) * float(x[ii])
                    red[off_b1 + j] -= lr * float(delta_h[j])

        return red

    def mlp_predecir(self, red, X_plano, n):
        """Predicción MLP: retorna salidas"""
        n = int(n)
        n_in = int(red[0])
        n_out = int(red[2])

        if not isinstance(X_plano, list):
            raise RuntimeError("mlp_predecir() espera arreglo X_plano")

        preds = []
        for i in range(n):
            x = []
            for j in range(n_in):
                x.append(float(X_plano[i * n_in + j]))
            act = self.mlp_forward(red, x, n_in)
            for j in range(n_out):
                preds.append(float(act[n_in + j]))

        return preds

    # ─── K-Means clustering ───────────────────────────────────

    def _distancia_euclid(self, X_plano, i, centroides, c, m):
        """Distancia euclidiana entre punto i y centroide c"""
        m = int(m)
        suma = 0.0
        for j in range(m):
            d = float(X_plano[i * m + j]) - float(centroides[c * m + j])
            suma += d * d
        return suma

    def kmeans(self, X_plano, n, m, k, max_iter):
        """
        K-Means clustering.
        X_plano: n x m (datos planos)
        k: número de clusters
        max_iter: iteraciones máximas
        Retorna etiquetas [0..k-1] para cada punto
        """
        n = int(n)
        m = int(m)
        k = int(k)
        max_iter = int(max_iter)

        if not isinstance(X_plano, list):
            raise RuntimeError("kmeans() espera arreglo X_plano")

        # Inicializar centroides con primeros k puntos
        centroides = []
        for c in range(k):
            for j in range(m):
                centroides.append(float(X_plano[c * m + j]))

        # Inicializar etiquetas
        etiquetas = [0] * n

        for iter in range(max_iter):
            # Paso E: asignar puntos
            cambios = 0
            for i in range(n):
                mejor_c = 0
                mejor_d = self._distancia_euclid(X_plano, i, centroides, 0, m)
                for c in range(1, k):
                    d = self._distancia_euclid(X_plano, i, centroides, c, m)
                    if d < mejor_d:
                        mejor_d = d
                        mejor_c = c
                if etiquetas[i] != mejor_c:
                    cambios += 1
                etiquetas[i] = mejor_c

            # Paso M: recalcular centroides
            sumas = [0.0] * (k * m)
            conteos = [0] * k

            for i in range(n):
                c = etiquetas[i]
                for j in range(m):
                    sumas[c * m + j] += float(X_plano[i * m + j])
                conteos[c] += 1

            for c in range(k):
                if conteos[c] > 0:
                    for j in range(m):
                        centroides[c * m + j] = sumas[c * m + j] / conteos[c]

            if cambios == 0:
                break

        return etiquetas

    def kmeans_resumen(self, etiquetas, n, k):
        """Imprime resumen de clusters"""
        n = int(n)
        k = int(k)
        if not isinstance(etiquetas, list):
            raise RuntimeError("kmeans_resumen() espera arreglo etiquetas")

        print("--- Resumen K-Means ---")
        for c in range(k):
            count = 0
            for i in range(n):
                if int(etiquetas[i]) == c:
                    count += 1
            print(f"Cluster {c}: {count} puntos")

        return 0
