"""Librería estándar de DeepLang para lectura y escritura de archivos."""


class DeepLangFileLib:
    """Funciones de archivos para integrarse como builtins."""

    # ─── Lectura ─────────────────────────────────────────────
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

    # ─── Escritura ───────────────────────────────────────────
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
