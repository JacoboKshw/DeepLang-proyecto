"""Librería estándar de DeepLang para lectura y escritura de archivos."""


class DeepLangFileLib:
    """Funciones de archivos para integrarse como builtins."""

    # ─── Lectura de texto ────────────────────────────────────
    def leerarchivo(self, ruta):
        ruta_str = self._validar_ruta(ruta)
        try:
            with open(ruta_str, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError as exc:
            raise RuntimeError(f"No existe el archivo: {ruta_str}") from exc
        except IsADirectoryError as exc:
            raise RuntimeError(f"La ruta apunta a un directorio: {ruta_str}") from exc
        except PermissionError as exc:
            raise RuntimeError(f"Sin permisos para leer: {ruta_str}") from exc
        except OSError as exc:
            raise RuntimeError(f"Error al leer '{ruta_str}': {exc}") from exc

    def leerlineas(self, ruta):
        return self.leerarchivo(ruta).splitlines()

    # ─── Escritura de texto ──────────────────────────────────
    def escribirarchivo(self, ruta, contenido):
        """Crea o sobreescribe el archivo con el texto dado."""
        ruta_str = self._validar_ruta(ruta)
        texto    = self._validar_texto(contenido)
        try:
            with open(ruta_str, "w", encoding="utf-8") as f:
                f.write(texto)
            return 1
        except PermissionError as exc:
            raise RuntimeError(f"Sin permisos para escribir: {ruta_str}") from exc
        except OSError as exc:
            raise RuntimeError(f"Error al escribir '{ruta_str}': {exc}") from exc

    def agregararchivo(self, ruta, contenido):
        """Agrega texto al final del archivo (append)."""
        ruta_str = self._validar_ruta(ruta)
        texto    = self._validar_texto(contenido)
        try:
            with open(ruta_str, "a", encoding="utf-8") as f:
                f.write(texto)
            return 1
        except PermissionError as exc:
            raise RuntimeError(f"Sin permisos para escribir: {ruta_str}") from exc
        except OSError as exc:
            raise RuntimeError(f"Error al agregar a '{ruta_str}': {exc}") from exc

    # ─── Lectura de CSV ──────────────────────────────────────
    def leercsv(self, ruta):
        """
        Lee un CSV y devuelve una lista de listas (arreglo de filas).
        Cada fila es un arreglo de valores: int, float o string.
        La primera fila (encabezado) se incluye como fila de strings.

        Ejemplo de uso en DeepLang:
            tabla = leercsv("datos.csv")
            print(tabla[0])        # primera fila (encabezados)
            print(tabla[1])        # segunda fila (primer dato)
            print(tabla[1][0])     # celda fila 1, columna 0
        """
        lineas = self.leerlineas(ruta)
        if not lineas:
            return []
        resultado = []
        for linea in lineas:
            if not linea.strip():          # ignorar líneas vacías
                continue
            celdas = self._parsear_fila_csv(linea)
            resultado.append(celdas)
        return resultado

    def leercsv_datos(self, ruta):
        """
        Lee un CSV saltando la primera fila (encabezados).
        Solo devuelve filas de datos, convirtiendo números automáticamente.

        Ejemplo de uso en DeepLang:
            datos = leercsv_datos("datos.csv")
            print(datos[0][0])     # primera celda del primer dato
        """
        tabla = self.leercsv(ruta)
        if len(tabla) <= 1:
            return []
        return tabla[1:]           # saltamos el encabezado

    def leercsv_columna(self, ruta, col):
        """
        Lee una columna específica de un CSV (índice base 0),
        saltando el encabezado. Devuelve un arreglo de valores.

        Ejemplo de uso en DeepLang:
            edades = leercsv_columna("datos.csv", 1)
            print(edades[0])
        """
        col = int(col)
        datos = self.leercsv_datos(ruta)
        columna = []
        for fila in datos:
            if col < len(fila):
                columna.append(fila[col])
            else:
                raise RuntimeError(
                    f"La columna {col} no existe en '{ruta}' "
                    f"(la fila solo tiene {len(fila)} columnas)"
                )
        return columna

    def escribircsv(self, ruta, tabla):
        """
        Escribe una tabla (arreglo de arreglos) como archivo CSV.
        Crea o sobreescribe el archivo.

        Ejemplo de uso en DeepLang:
            fila0 = ["nombre", "edad", "nota"]
            fila1 = ["Ana", 20, 9.5]
            tabla = [fila0, fila1]
            escribircsv("salida.csv", tabla)
        """
        ruta_str = self._validar_ruta(ruta)
        if not isinstance(tabla, list):
            raise RuntimeError("escribircsv() espera un arreglo de filas")
        lineas = []
        for fila in tabla:
            if not isinstance(fila, list):
                raise RuntimeError("Cada fila del CSV debe ser un arreglo")
            celdas = []
            for celda in fila:
                celdas.append(self._formatear_celda_csv(celda))
            lineas.append(",".join(celdas))
        contenido = "\n".join(lineas) + "\n"
        try:
            with open(ruta_str, "w", encoding="utf-8") as f:
                f.write(contenido)
            return 1
        except PermissionError as exc:
            raise RuntimeError(f"Sin permisos para escribir: {ruta_str}") from exc
        except OSError as exc:
            raise RuntimeError(f"Error al escribir '{ruta_str}': {exc}") from exc

    # ─── Helpers CSV ─────────────────────────────────────────
    def _parsear_fila_csv(self, linea):
        """
        Parsea una línea CSV respetando comillas dobles.
        Convierte automáticamente a int o float cuando es posible.
        """
        celdas  = []
        celda   = []
        en_comillas = False
        i = 0
        while i < len(linea):
            c = linea[i]
            if c == '"':
                if en_comillas and i + 1 < len(linea) and linea[i + 1] == '"':
                    # Comilla escapada dentro de campo entre comillas: ""
                    celda.append('"')
                    i += 2
                    continue
                en_comillas = not en_comillas
            elif c == ',' and not en_comillas:
                celdas.append(self._convertir_celda((''.join(celda)).strip()))
                celda = []
            else:
                celda.append(c)
            i += 1
        # Última celda
        celdas.append(self._convertir_celda((''.join(celda)).strip()))
        return celdas

    @staticmethod
    def _convertir_celda(texto):
        """Convierte una celda a int, float o string según corresponda."""
        # Quitar comillas externas si las tiene
        if texto.startswith('"') and texto.endswith('"'):
            return texto[1:-1]
        # Intentar int
        try:
            return int(texto)
        except ValueError:
            pass
        # Intentar float
        try:
            return float(texto)
        except ValueError:
            pass
        # Dejar como string
        return texto

    @staticmethod
    def _formatear_celda_csv(valor):
        """Formatea un valor de DeepLang para escribirlo en CSV."""
        if isinstance(valor, str):
            # Si contiene coma, salto de línea o comilla, encerrar entre comillas
            if ',' in valor or '"' in valor or '\n' in valor:
                return '"' + valor.replace('"', '""') + '"'
            return valor
        if isinstance(valor, float) and valor == int(valor):
            return str(int(valor))
        return str(valor)

    # ─── Validadores ─────────────────────────────────────────
    @staticmethod
    def _validar_ruta(ruta):
        if not isinstance(ruta, str):
            raise RuntimeError("La ruta del archivo debe ser texto")
        ruta = ruta.strip()
        if not ruta:
            raise RuntimeError("La ruta del archivo no puede estar vacía")
        return ruta

    @staticmethod
    def _validar_texto(contenido):
        if isinstance(contenido, str):
            return contenido
        if isinstance(contenido, (int, float)):
            return str(contenido)
        raise RuntimeError("El contenido a escribir debe ser texto o número")
