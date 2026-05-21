"""Librería estándar de DeepLang para gráficas ASCII."""


class DeepLangGraficasLib:
    """
    Gráficas ASCII para integrarse como builtins en DeepLang.

    Funciones disponibles:
        grafica_barras(datos, n, ancho)
        grafica_barras_v(datos, n, alto)
        grafica_linea(datos, n, ancho, alto)
        grafica_dispersion(X, Y, n, w, h)
        histograma(datos, n, bins, ancho)
        grafica_pastel(datos, n)
        grafica_funcion(Y, n, xmin, xmax, ancho, alto)
    """

    # ─── Helpers internos ────────────────────────────────────

    @staticmethod
    def _validar_arreglo(arr, nombre="datos"):
        if not isinstance(arr, list) or len(arr) == 0:
            raise RuntimeError(f"'{nombre}' debe ser un arreglo no vacío")

    @staticmethod
    def _max_arr(arr, n):
        m = arr[0]
        for i in range(1, n):
            if arr[i] > m:
                m = arr[i]
        return m

    @staticmethod
    def _min_arr(arr, n):
        m = arr[0]
        for i in range(1, n):
            if arr[i] < m:
                m = arr[i]
        return m

    @staticmethod
    def _repetir_char(c, veces):
        return c * max(0, int(veces))

    @staticmethod
    def _redondear_entero(x):
        if x >= 0:
            return int(x + 0.5)
        return int(x - 0.5)

    def _fmt(self, v):
        """Formatea un número igual que EvalVisitor.visitPrint."""
        if isinstance(v, float) and v == int(v):
            return str(int(v))
        if isinstance(v, float):
            return f"{v:.10f}".rstrip("0").rstrip(".")
        return str(v)

    # ─── grafica_barras ──────────────────────────────────────

    def grafica_barras(self, datos, n, ancho):
        """Barras horizontales. ancho = máximo de caracteres por barra."""
        self._validar_arreglo(datos)
        n    = int(n)
        ancho = int(ancho)
        mx   = self._max_arr(datos, n)
        if mx == 0:
            mx = 1.0
        print("┌─ Gráfica de barras ────────────────────────────────")
        for i in range(n):
            val       = datos[i]
            barra_len = max(0, self._redondear_entero(val * ancho / mx))
            barra     = self._repetir_char("█", barra_len)
            print(f"{barra}  {self._fmt(val)}")
        print("└────────────────────────────────────────────────────")
        return 0

    # ─── grafica_barras_v ────────────────────────────────────

    def grafica_barras_v(self, datos, n, alto):
        """Barras verticales. alto = altura en filas."""
        self._validar_arreglo(datos)
        n    = int(n)
        alto = int(alto)
        mx   = self._max_arr(datos, n)
        if mx == 0:
            mx = 1.0
        print("┌─ Barras verticales ────────────────────────────────")
        for fila in range(alto, 0, -1):
            linea = ""
            for i in range(n):
                umbral = fila * mx / alto
                if datos[i] >= umbral:
                    linea += "█ "
                else:
                    linea += "  "
            print(linea)
        print(self._repetir_char("▔▔", n))
        etiquetas = ""
        for i in range(n):
            etiquetas += str(i) + " "
        print(etiquetas)
        print("└────────────────────────────────────────────────────")
        return 0

    # ─── grafica_linea ───────────────────────────────────────

    def grafica_linea(self, datos, n, ancho, alto):
        """Gráfica de línea ASCII. ancho = columnas, alto = filas."""
        self._validar_arreglo(datos)
        n     = int(n)
        ancho = int(ancho)
        alto  = int(alto)
        mn    = self._min_arr(datos, n)
        mx    = self._max_arr(datos, n)
        rango = mx - mn
        if abs(rango) < 1e-10:
            rango = 1.0
        print("┌─ Gráfica de línea ─────────────────────────────────")
        for fila in range(alto - 1, -1, -1):
            linea = "|"
            for col in range(ancho):
                idx = self._redondear_entero(col * (n - 1) / ancho)
                idx = min(idx, n - 1)
                val   = datos[idx]
                y_pos = self._redondear_entero((val - mn) / rango * (alto - 1))
                linea += "●" if y_pos == fila else " "
            print(linea)
        print("+" + self._repetir_char("-", ancho))
        print(f"  min={self._fmt(mn)}  max={self._fmt(mx)}")
        print("└────────────────────────────────────────────────────")
        return 0

    # ─── grafica_dispersion ──────────────────────────────────

    def grafica_dispersion(self, X, Y, n, w, h):
        """Diagrama de dispersión X vs Y."""
        self._validar_arreglo(X, "X")
        self._validar_arreglo(Y, "Y")
        n = int(n)
        w = int(w)
        h = int(h)
        mn_x = self._min_arr(X, n)
        mx_x = self._max_arr(X, n)
        mn_y = self._min_arr(Y, n)
        mx_y = self._max_arr(Y, n)
        rng_x = mx_x - mn_x if abs(mx_x - mn_x) >= 1e-10 else 1.0
        rng_y = mx_y - mn_y if abs(mx_y - mn_y) >= 1e-10 else 1.0

        canvas = [0] * (w * h)
        for i in range(n):
            cx = self._redondear_entero((X[i] - mn_x) / rng_x * (w - 1))
            cy = self._redondear_entero((Y[i] - mn_y) / rng_y * (h - 1))
            cx = max(0, min(cx, w - 1))
            cy = max(0, min(cy, h - 1))
            canvas[(h - 1 - cy) * w + cx] = 1

        print("┌─ Dispersión ───────────────────────────────────────")
        for fila in range(h):
            linea = "|"
            for col in range(w):
                linea += "·" if canvas[fila * w + col] == 1 else " "
            print(linea)
        print("+" + self._repetir_char("-", w))
        print(f"  X: {self._fmt(mn_x)} .. {self._fmt(mx_x)}")
        print(f"  Y: {self._fmt(mn_y)} .. {self._fmt(mx_y)}")
        print("└────────────────────────────────────────────────────")
        return 0

    # ─── histograma ──────────────────────────────────────────

    def histograma(self, datos, n, bins, ancho):
        """Histograma de frecuencias."""
        self._validar_arreglo(datos)
        n     = int(n)
        bins  = int(bins)
        ancho = int(ancho)
        mn    = self._min_arr(datos, n)
        mx    = self._max_arr(datos, n)
        rng   = mx - mn
        if abs(rng) < 1e-10:
            rng = 1.0

        freqs = [0] * bins
        for i in range(n):
            b = self._redondear_entero((datos[i] - mn) / rng * (bins - 1))
            b = max(0, min(b, bins - 1))
            freqs[b] += 1

        mx_f = max(freqs) if max(freqs) > 0 else 1

        print("┌─ Histograma ───────────────────────────────────────")
        for b in range(bins):
            limite_inf = mn + b * rng / bins
            # redondear a 2 decimales
            factor     = 100.0
            if limite_inf >= 0:
                limite_inf = int(limite_inf * factor + 0.5) / factor
            else:
                limite_inf = -int(-limite_inf * factor + 0.5) / factor
            barra_len  = self._redondear_entero(freqs[b] * ancho / mx_f)
            barra      = self._repetir_char("▓", barra_len)
            print(f"{self._fmt(limite_inf)} | {barra} {freqs[b]}")
        print("└────────────────────────────────────────────────────")
        return 0

    # ─── grafica_pastel ──────────────────────────────────────

    def grafica_pastel(self, datos, n):
        """Gráfica de pastel en texto (proporciones con barras)."""
        self._validar_arreglo(datos)
        n    = int(n)
        suma = sum(datos[:n])
        if suma < 1e-10:
            print("Error: todos los valores son cero")
            return 0
        print("┌─ Gráfica de pastel ────────────────────────────────")
        for i in range(n):
            pct    = datos[i] * 100.0 / suma
            # redondear pct a 1 decimal
            pct_r  = int(pct * 10 + 0.5) / 10.0
            barras = self._redondear_entero(datos[i] * 40 / suma)
            barra  = self._repetir_char("█", barras)
            print(f"[{i}] {barra} {self._fmt(pct_r)}%  ({self._fmt(datos[i])})")
        print("└────────────────────────────────────────────────────")
        return 0

    # ─── grafica_funcion ─────────────────────────────────────

    def grafica_funcion(self, Y, n, xmin, xmax, ancho, alto):
        """
        Gráfica de función y = f(x).
        Y: arreglo de muestras uniformes en [xmin, xmax].
        """
        self._validar_arreglo(Y, "Y")
        n     = int(n)
        ancho = int(ancho)
        alto  = int(alto)
        mn    = self._min_arr(Y, n)
        mx    = self._max_arr(Y, n)
        rng   = mx - mn
        if abs(rng) < 1e-10:
            rng = 1.0

        print("┌─ Gráfica de función ───────────────────────────────")
        for fila in range(alto - 1, -1, -1):
            linea = "|"
            for col in range(ancho):
                idx = self._redondear_entero(col * (n - 1) / (ancho - 1))
                idx = min(idx, n - 1)
                val   = Y[idx]
                y_pos = self._redondear_entero((val - mn) / rng * (alto - 1))
                linea += "*" if y_pos == fila else " "
            print(linea)
        print("+" + self._repetir_char("-", ancho))
        # redondear mn y mx a 3 decimales para mostrar
        def _r3(v):
            f = 1000.0
            return (int(v * f + 0.5) / f) if v >= 0 else (-int(-v * f + 0.5) / f)
        print(f"  x: {self._fmt(xmin)} .. {self._fmt(xmax)}"
              f"   y: {self._fmt(_r3(mn))} .. {self._fmt(_r3(mx))}")
        print("└────────────────────────────────────────────────────")
        return 0
