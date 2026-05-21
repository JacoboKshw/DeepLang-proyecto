"""Librería estándar de DeepLang para operaciones matriciales."""


class DeepLangMatricesLib:
    """
    Matrices representadas como arreglos planos (lista de listas en Python).
    Convención: elemento [i][j] está en índice i*cols + j.

    Funciones disponibles:
        mat_get(M, cols, fila, col)
        mat_idx(cols, fila, col)
        mat_ceros(filas, cols)
        mat_identidad(n)
        mat_suma(A, B, filas, cols)
        mat_resta(A, B, filas, cols)
        mat_escalar(A, filas, cols, k)
        mat_mul(A, B, m, n, p)
        mat_transpuesta(A, filas, cols)
        mat_traza(A, n)
        mat_inversa(A, n)
        mat_det(A, n)
        mat_imprimir(A, filas, cols)
        vec_dot(u, v, n)
        vec_norma(v, n)
    """

    # ─── Validadores ─────────────────────────────────────────

    @staticmethod
    def _validar_matriz(M, nombre="M"):
        if not isinstance(M, list):
            raise RuntimeError(f"'{nombre}' debe ser un arreglo")

    @staticmethod
    def _validar_dims(*args):
        for v in args:
            if int(v) <= 0:
                raise RuntimeError("Las dimensiones deben ser enteros positivos")

    # ─── Acceso y construcción ───────────────────────────────

    def mat_get(self, M, cols, fila, col):
        """Obtiene el elemento M[fila][col] de la matriz plana."""
        self._validar_matriz(M)
        return M[int(fila) * int(cols) + int(col)]

    def mat_idx(self, cols, fila, col):
        """Devuelve el índice plano de M[fila][col]."""
        return int(fila) * int(cols) + int(col)

    def mat_ceros(self, filas, cols):
        """Crea una matriz de filas×cols llena de ceros."""
        self._validar_dims(filas, cols)
        return [0.0] * (int(filas) * int(cols))

    def mat_identidad(self, n):
        """Crea la matriz identidad n×n."""
        self._validar_dims(n)
        n   = int(n)
        M   = [0.0] * (n * n)
        for i in range(n):
            M[i * n + i] = 1.0
        return M

    # ─── Operaciones básicas ─────────────────────────────────

    def mat_suma(self, A, B, filas, cols):
        """A + B (mismas dimensiones filas×cols)."""
        self._validar_matriz(A, "A")
        self._validar_matriz(B, "B")
        total = int(filas) * int(cols)
        return [A[i] + B[i] for i in range(total)]

    def mat_resta(self, A, B, filas, cols):
        """A - B (mismas dimensiones filas×cols)."""
        self._validar_matriz(A, "A")
        self._validar_matriz(B, "B")
        total = int(filas) * int(cols)
        return [A[i] - B[i] for i in range(total)]

    def mat_escalar(self, A, filas, cols, k):
        """Multiplica cada elemento de A por el escalar k."""
        self._validar_matriz(A)
        total = int(filas) * int(cols)
        return [A[i] * k for i in range(total)]

    def mat_mul(self, A, B, m, n, p):
        """
        Multiplicación de matrices A (m×n) * B (n×p) = C (m×p).
        """
        self._validar_matriz(A, "A")
        self._validar_matriz(B, "B")
        m, n, p = int(m), int(n), int(p)
        C = [0.0] * (m * p)
        for i in range(m):
            for j in range(p):
                s = 0.0
                for k in range(n):
                    s += A[i * n + k] * B[k * p + j]
                C[i * p + j] = s
        return C

    def mat_transpuesta(self, A, filas, cols):
        """Transpuesta de A (filas×cols) → resultado (cols×filas)."""
        self._validar_matriz(A)
        filas, cols = int(filas), int(cols)
        T = [0.0] * (filas * cols)
        for i in range(filas):
            for j in range(cols):
                T[j * filas + i] = A[i * cols + j]
        return T

    def mat_traza(self, A, n):
        """Suma de la diagonal principal de la matriz cuadrada n×n."""
        self._validar_matriz(A)
        n = int(n)
        return sum(A[i * n + i] for i in range(n))

    # ─── Impresión ───────────────────────────────────────────

    def mat_imprimir(self, A, filas, cols):
        """Imprime la matriz con formato legible."""
        self._validar_matriz(A)
        filas, cols = int(filas), int(cols)
        for i in range(filas):
            fila_str = "[ "
            for j in range(cols):
                val = A[i * cols + j]
                if isinstance(val, float) and val == int(val):
                    fila_str += str(int(val))
                elif isinstance(val, float):
                    fila_str += f"{val:.6f}".rstrip("0").rstrip(".")
                else:
                    fila_str += str(val)
                if j < cols - 1:
                    fila_str += "  "
            fila_str += " ]"
            print(fila_str)
        return 0

    # ─── Inversa (Gauss-Jordan con pivoteo parcial) ──────────

    def mat_inversa(self, A, n):
        """
        Inversa de una matriz cuadrada n×n.
        Usa eliminación de Gauss-Jordan con pivoteo parcial.
        Retorna [] si la matriz es singular.
        """
        self._validar_matriz(A)
        n   = int(n)
        Aug = list(A)          # copia de A
        Inv = [0.0] * (n * n)
        for i in range(n):
            Inv[i * n + i] = 1.0

        for col in range(n):
            # Pivoteo parcial
            pivot_fila = col
            pivot_val  = abs(Aug[col * n + col])
            for fila in range(col + 1, n):
                v = abs(Aug[fila * n + col])
                if v > pivot_val:
                    pivot_val  = v
                    pivot_fila = fila

            if pivot_val < 1e-12:
                print("Error: la matriz es singular (no tiene inversa)")
                return []

            # Intercambiar filas
            if pivot_fila != col:
                for j in range(n):
                    Aug[col * n + j],        Aug[pivot_fila * n + j]        = \
                        Aug[pivot_fila * n + j], Aug[col * n + j]
                    Inv[col * n + j],        Inv[pivot_fila * n + j]        = \
                        Inv[pivot_fila * n + j], Inv[col * n + j]

            # Escalar fila pivot
            pivote = Aug[col * n + col]
            for j in range(n):
                Aug[col * n + j] /= pivote
                Inv[col * n + j] /= pivote

            # Eliminar arriba y abajo
            for fila in range(n):
                if fila != col:
                    factor = Aug[fila * n + col]
                    for j in range(n):
                        Aug[fila * n + j] -= factor * Aug[col * n + j]
                        Inv[fila * n + j] -= factor * Inv[col * n + j]

        return Inv

    # ─── Determinante (eliminación gaussiana) ────────────────

    def mat_det(self, A, n):
        """
        Determinante de una matriz cuadrada n×n.
        Usa eliminación gaussiana con pivoteo parcial.
        """
        self._validar_matriz(A)
        n   = int(n)
        M   = list(A)
        det = 1.0
        sig = 1.0

        for col in range(n):
            pivot_fila = col
            pivot_val  = abs(M[col * n + col])
            for fila in range(col + 1, n):
                v = abs(M[fila * n + col])
                if v > pivot_val:
                    pivot_val  = v
                    pivot_fila = fila

            if pivot_val < 1e-12:
                return 0.0

            if pivot_fila != col:
                for j in range(n):
                    M[col * n + j], M[pivot_fila * n + j] = \
                        M[pivot_fila * n + j], M[col * n + j]
                sig *= -1.0

            pivote = M[col * n + col]
            det   *= pivote
            for fila in range(col + 1, n):
                factor = M[fila * n + col] / pivote
                for j in range(col, n):
                    M[fila * n + j] -= factor * M[col * n + j]

        return det * sig

    # ─── Vectores ────────────────────────────────────────────

    def vec_dot(self, u, v, n):
        """Producto punto de dos vectores de longitud n."""
        self._validar_matriz(u, "u")
        self._validar_matriz(v, "v")
        n = int(n)
        return sum(u[i] * v[i] for i in range(n))

    def vec_norma(self, v, n):
        """Norma euclidiana de un vector de longitud n."""
        dot = self.vec_dot(v, v, n)
        if dot == 0:
            return 0.0
        guess = float(dot) if dot >= 1 else 1.0
        for _ in range(40):
            guess = 0.5 * (guess + dot / guess)
        return guess
