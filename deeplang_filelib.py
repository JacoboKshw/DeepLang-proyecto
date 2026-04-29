"""Librería estándar de DeepLang para lectura de archivos.
"""

class DeepLangFileLib:
    """Funciones de lectura de archivos para integrarse como builtins."""

    def leerarchivo(self, ruta):
        ruta_str = self._validar_ruta(ruta)
        try:
            with open(ruta_str, "r", encoding="utf-8") as archivo:
                return archivo.read()
        except FileNotFoundError as exc:
            raise RuntimeError(f"No existe el archivo: {ruta_str}") from exc
        except IsADirectoryError as exc:
            raise RuntimeError(f"La ruta apunta a un directorio: {ruta_str}") from exc
        except PermissionError as exc:
            raise RuntimeError(f"Sin permisos para leer: {ruta_str}") from exc
        except OSError as exc:
            raise RuntimeError(f"Error al leer '{ruta_str}': {exc}") from exc

    def leerlineas(self, ruta):
        contenido = self.leerarchivo(ruta)
        return contenido.splitlines()

    @staticmethod
    def _validar_ruta(ruta):
        if not isinstance(ruta, str):
            raise RuntimeError("La ruta del archivo debe ser texto")
        ruta = ruta.strip()
        if not ruta:
            raise RuntimeError("La ruta del archivo no puede estar vacía")
        return ruta
