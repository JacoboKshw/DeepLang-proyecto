"""Librería estándar de DeepLang para lectura y escritura de archivos."""

import zipfile
import xml.etree.ElementTree as ET


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
        lineas = self.leerlineas(ruta)
        if not lineas:
            return []
        resultado = []
        for linea in lineas:
            if not linea.strip():
                continue
            celdas = self._parsear_fila_csv(linea)
            resultado.append(celdas)
        return resultado

    def leercsv_datos(self, ruta):
        tabla = self.leercsv(ruta)
        if len(tabla) <= 1:
            return []
        return tabla[1:]

    def leercsv_columna(self, ruta, col):
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

    # ─── Lectura de XLSX (Python puro) ───────────────────────

    def _xlsx_cargar(self, ruta):
        """
        Abre un .xlsx y devuelve (sheets_dict, shared_strings).
        sheets_dict = { nombre_hoja: xml_bytes }
        shared_strings = [ str, ... ]
        """
        ruta_str = self._validar_ruta(ruta)
        try:
            zf = zipfile.ZipFile(ruta_str, "r")
        except FileNotFoundError as exc:
            raise RuntimeError(f"No existe el archivo: {ruta_str}") from exc
        except zipfile.BadZipFile as exc:
            raise RuntimeError(f"El archivo no es un xlsx válido: {ruta_str}") from exc

        nombres = zf.namelist()

        # ── Shared strings ────────────────────────────────────
        shared = []
        if "xl/sharedStrings.xml" in nombres:
            ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall(f"{{{ns}}}si"):
                partes = []
                # Texto simple <t>
                t = si.find(f"{{{ns}}}t")
                if t is not None and t.text:
                    partes.append(t.text)
                # Texto enriquecido <r><t>
                for r in si.findall(f"{{{ns}}}r"):
                    rt = r.find(f"{{{ns}}}t")
                    if rt is not None and rt.text:
                        partes.append(rt.text)
                shared.append("".join(partes))

        # ── Nombres de hojas ──────────────────────────────────
        wb_ns  = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
        wb_root = ET.fromstring(zf.read("xl/workbook.xml"))
        sheets_info = {}   # nombre -> rId
        for sh in wb_root.find(f"{{{wb_ns}}}sheets").findall(f"{{{wb_ns}}}sheet"):
            r_ns  = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
            rid   = sh.get(f"{{{r_ns}}}id")
            sname = sh.get("name")
            sheets_info[sname] = rid

        # ── Relaciones workbook ───────────────────────────────
        rel_root = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_ns   = "http://schemas.openxmlformats.org/package/2006/relationships"
        rid_to_path = {}
        for rel in rel_root.findall(f"{{{rel_ns}}}Relationship"):
            rid_to_path[rel.get("Id")] = "xl/" + rel.get("Target").lstrip("/").replace("xl/", "")

        sheets_xml = {}
        for sname, rid in sheets_info.items():
            path = rid_to_path.get(rid)
            if path and path in nombres:
                sheets_xml[sname] = zf.read(path)

        zf.close()
        return sheets_xml, shared

    def _xlsx_parsear_hoja(self, xml_bytes, shared):
        """
        Parsea el XML de una hoja y devuelve lista de listas (tabla).
        Respeta celdas vacías usando la referencia de columna.
        """
        ns   = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
        root = ET.fromstring(xml_bytes)
        sd   = root.find(f"{{{ns}}}sheetData")
        if sd is None:
            return []

        tabla = []
        for row_el in sd.findall(f"{{{ns}}}row"):
            fila = {}
            for c_el in row_el.findall(f"{{{ns}}}c"):
                ref  = c_el.get("r", "")          # ej. "A1", "C3"
                tipo = c_el.get("t", "")           # "s"=shared, "n"=number, ""=number
                v_el = c_el.find(f"{{{ns}}}v")
                val  = v_el.text if v_el is not None else ""

                # Convertir valor
                if tipo == "s":
                    # Cadena compartida
                    idx = int(val) if val else 0
                    val = shared[idx] if idx < len(shared) else ""
                elif val == "" or val is None:
                    val = ""
                else:
                    # Número: int o float
                    try:
                        f = float(val)
                        val = int(f) if f == int(f) else f
                    except ValueError:
                        pass

                # Obtener índice de columna a partir de la referencia
                col_idx = self._ref_a_col(ref)
                fila[col_idx] = val

            if fila:
                max_col = max(fila.keys()) + 1
                tabla.append([fila.get(i, "") for i in range(max_col)])

        return tabla

    @staticmethod
    def _ref_a_col(ref):
        """Convierte referencia como 'A1', 'BC3' al índice de columna (0-based)."""
        col = 0
        for ch in ref:
            if ch.isalpha():
                col = col * 26 + (ord(ch.upper()) - ord("A") + 1)
            else:
                break
        return col - 1

    def leerxlsx(self, ruta):
        """
        Lee la primera hoja de un archivo .xlsx.
        Retorna tabla como arreglo de arreglos (incluyendo encabezado).
        """
        sheets, shared = self._xlsx_cargar(ruta)
        if not sheets:
            raise RuntimeError(f"El archivo no contiene hojas: {ruta}")
        primera = next(iter(sheets.values()))
        return self._xlsx_parsear_hoja(primera, shared)

    def leerxlsx_datos(self, ruta):
        """
        Lee la primera hoja de un .xlsx, omitiendo la primera fila (encabezado).
        """
        tabla = self.leerxlsx(ruta)
        return tabla[1:] if len(tabla) > 1 else []

    def leerxlsx_hoja(self, ruta, nombre_hoja):
        """
        Lee una hoja específica de un .xlsx por nombre.
        Retorna tabla como arreglo de arreglos.
        """
        sheets, shared = self._xlsx_cargar(ruta)
        nombre_hoja = str(nombre_hoja)
        if nombre_hoja not in sheets:
            disponibles = ", ".join(sheets.keys())
            raise RuntimeError(
                f"La hoja '{nombre_hoja}' no existe en '{ruta}'. "
                f"Hojas disponibles: {disponibles}"
            )
        return self._xlsx_parsear_hoja(sheets[nombre_hoja], shared)

    def leerxlsx_columna(self, ruta, col):
        """
        Lee una columna específica (índice 0-based) de la primera hoja,
        omitiendo el encabezado.
        """
        col = int(col)
        datos = self.leerxlsx_datos(ruta)
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

    def leerxlsx_hojas(self, ruta):
        """
        Retorna un arreglo con los nombres de todas las hojas del archivo.
        """
        sheets, _ = self._xlsx_cargar(ruta)
        return list(sheets.keys())

    # ─── Helpers CSV ─────────────────────────────────────────
    def _parsear_fila_csv(self, linea):
        celdas  = []
        celda   = []
        en_comillas = False
        i = 0
        while i < len(linea):
            c = linea[i]
            if c == '"':
                if en_comillas and i + 1 < len(linea) and linea[i + 1] == '"':
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
        celdas.append(self._convertir_celda((''.join(celda)).strip()))
        return celdas

    @staticmethod
    def _convertir_celda(texto):
        if texto.startswith('"') and texto.endswith('"'):
            return texto[1:-1]
        try:
            return int(texto)
        except ValueError:
            pass
        try:
            return float(texto)
        except ValueError:
            pass
        return texto

    @staticmethod
    def _formatear_celda_csv(valor):
        if isinstance(valor, str):
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
