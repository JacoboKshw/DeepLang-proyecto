"""Librería estándar de DeepLang para lectura y escritura de archivos.
"""


class DeepLangFileLib:
    """Funciones de archivos para integrarse como builtins."""

    # ═══════════════════════════════════════════════════════════
    # LECTURA DE TEXTO
    # ═══════════════════════════════════════════════════════════

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

    # ═══════════════════════════════════════════════════════════
    # ESCRITURA DE TEXTO
    # ═══════════════════════════════════════════════════════════

    def escribirarchivo(self, ruta, contenido):
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

    # ═══════════════════════════════════════════════════════════
    # CSV
    # ═══════════════════════════════════════════════════════════

    def leercsv(self, ruta):
        lineas = self.leerlineas(ruta)
        if not lineas:
            return []
        resultado = []
        for linea in lineas:
            if not linea.strip():
                continue
            resultado.append(self._parsear_fila_csv(linea))
        return resultado

    def leercsv_datos(self, ruta):
        tabla = self.leercsv(ruta)
        return tabla[1:] if len(tabla) > 1 else []

    def leercsv_columna(self, ruta, col):
        col   = int(col)
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
            lineas.append(",".join(self._formatear_celda_csv(c) for c in fila))
        contenido = "\n".join(lineas) + "\n"
        try:
            with open(ruta_str, "w", encoding="utf-8") as f:
                f.write(contenido)
            return 1
        except PermissionError as exc:
            raise RuntimeError(f"Sin permisos para escribir: {ruta_str}") from exc
        except OSError as exc:
            raise RuntimeError(f"Error al escribir '{ruta_str}': {exc}") from exc

    # ═══════════════════════════════════════════════════════════
    # ZIP PARSER
    # ═══════════════════════════════════════════════════════════

    def _zip_u16(self, d, p):
        return d[p] | (d[p+1] << 8)

    def _zip_u32(self, d, p):
        return d[p] | (d[p+1]<<8) | (d[p+2]<<16) | (d[p+3]<<24)

    def _zip_leer_archivos(self, ruta_str):
        """
        Lee un archivo ZIP sin compresión (method=0) y devuelve
        un diccionario { nombre: bytes }.
        """
        try:
            with open(ruta_str, "rb") as f:
                raw = f.read()
        except FileNotFoundError as exc:
            raise RuntimeError(f"No existe el archivo: {ruta_str}") from exc
        except OSError as exc:
            raise RuntimeError(f"Error al leer '{ruta_str}': {exc}") from exc

        # Firma de entrada local: PK\x03\x04
        SIG = bytes([0x50, 0x4B, 0x03, 0x04])
        resultado = {}
        pos = 0
        largo = len(raw)

        while pos < largo - 4:
            # Buscar siguiente firma
            idx = -1
            for i in range(pos, largo - 3):
                if raw[i]==0x50 and raw[i+1]==0x4B and raw[i+2]==0x03 and raw[i+3]==0x04:
                    idx = i
                    break
            if idx == -1:
                break

            if idx + 30 > largo:
                break

            method      = self._zip_u16(raw, idx + 8)
            comp_size   = self._zip_u32(raw, idx + 18)
            uncomp_size = self._zip_u32(raw, idx + 22)
            fname_len   = self._zip_u16(raw, idx + 26)
            extra_len   = self._zip_u16(raw, idx + 28)

            fname_bytes = raw[idx+30 : idx+30+fname_len]
            try:
                fname = fname_bytes.decode("utf-8")
            except Exception:
                fname = fname_bytes.decode("latin-1")

            data_start = idx + 30 + fname_len + extra_len
            data_end   = data_start + comp_size

            if method != 0:
                raise RuntimeError(
                    f"El archivo '{ruta_str}' usa compresión (method={method}) "
                    f"en la entrada '{fname}'. "
                    f"Guarda el Excel sin compresión para usar leerxlsx()."
                )

            resultado[fname] = raw[data_start:data_end]
            pos = data_end if data_end > idx + 1 else idx + 1

        if not resultado:
            raise RuntimeError(
                f"No se encontraron entradas ZIP en '{ruta_str}'. "
                f"¿Es un archivo .xlsx válido?"
            )

        return resultado

    # ═══════════════════════════════════════════════════════════
    # XML PARSER 
    # ═══════════════════════════════════════════════════════════

    def _xml_attrs(self, tag_texto):
        """
        Extrae atributos de un tag como string.
        Devuelve dict de atributos.
        """
        attrs  = {}
        i      = 0
        n      = len(tag_texto)
        # Saltar nombre del tag
        while i < n and tag_texto[i] not in (' ', '\t', '\n', '\r', '/', '>'):
            i += 1
        while i < n:
            # Saltar espacios
            while i < n and tag_texto[i] in (' ', '\t', '\n', '\r'):
                i += 1
            if i >= n or tag_texto[i] in ('/', '>'):
                break
            # Leer nombre de atributo
            j = i
            while j < n and tag_texto[j] not in ('=', ' ', '\t', '/', '>'):
                j += 1
            attr_name = tag_texto[i:j]
            i = j
            if i >= n or tag_texto[i] != '=':
                continue
            i += 1  # saltar '='
            if i >= n:
                break
            # Leer valor
            if tag_texto[i] in ('"', "'"):
                q = tag_texto[i]
                i += 1
                j = i
                while j < n and tag_texto[j] != q:
                    j += 1
                attr_val = tag_texto[i:j]
                i = j + 1
            else:
                j = i
                while j < n and tag_texto[j] not in (' ', '\t', '/', '>'):
                    j += 1
                attr_val = tag_texto[i:j]
                i = j
            if attr_name:
                # Quitar prefijo de namespace del nombre si lo tiene
                if ':' in attr_name:
                    attr_name = attr_name.split(':')[-1]
                attrs[attr_name] = self._xml_unescape(attr_val)
        return attrs

    def _xml_unescape(self, texto):
        """Reemplaza entidades XML básicas."""
        texto = texto.replace("&amp;",  "&")
        texto = texto.replace("&lt;",   "<")
        texto = texto.replace("&gt;",   ">")
        texto = texto.replace("&quot;", '"')
        texto = texto.replace("&apos;", "'")
        return texto

    def _xml_tag_name(self, tag_texto):
        """Extrae el nombre local del tag (sin namespace)."""
        i = 0
        n = len(tag_texto)
        while i < n and tag_texto[i] in ('/', '!', '?'):
            i += 1
        j = i
        while j < n and tag_texto[j] not in (' ', '\t', '\n', '\r', '/', '>'):
            j += 1
        nombre = tag_texto[i:j]
        if ':' in nombre:
            nombre = nombre.split(':')[-1]
        return nombre

    def _xml_tokens(self, texto):
        """
        Tokeniza XML en una lista de ('tag', contenido) o ('text', contenido).
        """
        tokens = []
        i      = 0
        n      = len(texto)
        while i < n:
            if texto[i] == '<':
                j = texto.find('>', i)
                if j == -1:
                    break
                tokens.append(('tag', texto[i+1:j]))
                i = j + 1
            else:
                j = texto.find('<', i)
                if j == -1:
                    j = n
                contenido = texto[i:j].strip()
                if contenido:
                    tokens.append(('text', self._xml_unescape(contenido)))
                i = j
        return tokens

    # ═══════════════════════════════════════════════════════════
    # PARSERS ESPECÍFICOS DE XLSX
    # ═══════════════════════════════════════════════════════════

    def _xlsx_shared_strings(self, xml_bytes):
        """
        Parsea sharedStrings.xml y devuelve lista de strings.
        Maneja tanto <t> simple como <r><t> (texto enriquecido).
        """
        texto  = xml_bytes.decode("utf-8", errors="replace")
        tokens = self._xml_tokens(texto)
        shared = []
        en_si  = False
        en_r   = False
        en_t   = False
        buf_si = []

        for tipo, contenido in tokens:
            if tipo == 'tag':
                nombre = self._xml_tag_name(contenido)
                cerrar = contenido.startswith('/')
                auto   = contenido.endswith('/')

                if not cerrar and nombre == 'si':
                    en_si  = True
                    buf_si = []
                elif cerrar and nombre == 'si':
                    shared.append("".join(buf_si))
                    en_si = False
                elif not cerrar and nombre == 'r' and en_si:
                    en_r = True
                elif cerrar and nombre == 'r':
                    en_r = False
                elif not cerrar and nombre == 't' and en_si:
                    en_t = True
                elif cerrar and nombre == 't':
                    en_t = False
            elif tipo == 'text' and en_si and en_t:
                buf_si.append(contenido)

        return shared

    def _xlsx_workbook_sheets(self, wb_bytes, rel_bytes):
        """
        Parsea workbook.xml y workbook.xml.rels.
        Devuelve lista de (nombre_hoja, ruta_relativa) en orden.
        """
        # ── Relaciones: rId -> target ─────────────────────────
        rel_texto  = rel_bytes.decode("utf-8", errors="replace")
        rid_target = {}
        for tipo, contenido in self._xml_tokens(rel_texto):
            if tipo == 'tag' and not contenido.startswith('/'):
                nombre = self._xml_tag_name(contenido)
                if nombre == 'Relationship':
                    attrs = self._xml_attrs(contenido)
                    rid   = attrs.get('Id', '')
                    tgt   = attrs.get('Target', '')
                    rid_target[rid] = tgt

        # ── Hojas en workbook.xml ─────────────────────────────
        wb_texto = wb_bytes.decode("utf-8", errors="replace")
        hojas    = []
        for tipo, contenido in self._xml_tokens(wb_texto):
            if tipo == 'tag' and not contenido.startswith('/'):
                nombre = self._xml_tag_name(contenido)
                if nombre == 'sheet':
                    attrs = self._xml_attrs(contenido)
                    sname = attrs.get('name', '')
                    rid   = attrs.get('id', '')
                    tgt   = rid_target.get(rid, '')
                    # Normalizar ruta: puede ser "worksheets/sheet1.xml" o "../worksheets/sheet1.xml"
                    tgt = tgt.lstrip('.').lstrip('/')
                    if not tgt.startswith('xl/'):
                        tgt = 'xl/' + tgt
                    hojas.append((sname, tgt))

        return hojas

    def _xlsx_parsear_hoja(self, xml_bytes, shared):
        """
        Parsea el XML de una hoja.
        Devuelve lista de listas respetando columnas vacías.
        """
        texto  = xml_bytes.decode("utf-8", errors="replace")
        tokens = self._xml_tokens(texto)

        tabla   = []
        en_row  = False
        en_c    = False
        en_v    = False
        tipo_c  = ''
        ref_c   = ''
        val_buf = []
        fila_actual = {}

        for tipo, contenido in tokens:
            if tipo == 'tag':
                nombre = self._xml_tag_name(contenido)
                cerrar = contenido.startswith('/')
                auto   = contenido.endswith('/')

                if not cerrar and nombre == 'row':
                    en_row     = True
                    fila_actual = {}

                elif cerrar and nombre == 'row':
                    if fila_actual:
                        max_col = max(fila_actual.keys()) + 1
                        tabla.append([fila_actual.get(i, "") for i in range(max_col)])
                    en_row = False

                elif not cerrar and nombre == 'c' and en_row:
                    attrs  = self._xml_attrs(contenido)
                    ref_c  = attrs.get('r', '')
                    tipo_c = attrs.get('t', '')
                    en_c   = True
                    val_buf = []
                    if auto:  # celda vacía self-closing
                        col_idx = self._ref_a_col(ref_c)
                        fila_actual[col_idx] = ""
                        en_c = False

                elif cerrar and nombre == 'c':
                    # Procesar valor acumulado
                    val_str = "".join(val_buf).strip()
                    if tipo_c == 's':
                        idx = int(val_str) if val_str else 0
                        val = shared[idx] if idx < len(shared) else ""
                    elif val_str == '':
                        val = ""
                    else:
                        try:
                            f = float(val_str)
                            val = int(f) if f == int(f) else f
                        except ValueError:
                            val = val_str
                    col_idx = self._ref_a_col(ref_c)
                    fila_actual[col_idx] = val
                    en_c  = False
                    en_v  = False

                elif not cerrar and nombre == 'v' and en_c:
                    en_v    = True
                    val_buf = []

                elif cerrar and nombre == 'v':
                    en_v = False

            elif tipo == 'text' and en_v:
                val_buf.append(contenido)

        return tabla

    @staticmethod
    def _ref_a_col(ref):
        """Convierte 'A1' -> 0, 'B3' -> 1, 'AA5' -> 26, etc."""
        col = 0
        for ch in ref:
            if ch.isalpha():
                col = col * 26 + (ord(ch.upper()) - ord('A') + 1)
            else:
                break
        return col - 1

    # ═══════════════════════════════════════════════════════════
    # API PÚBLICA XLSX
    # ═══════════════════════════════════════════════════════════

    def _xlsx_abrir(self, ruta):
        """
        Abre un xlsx sin compresión y devuelve (sheets_dict, shared_strings).
        sheets_dict = { nombre_hoja: xml_bytes }
        """
        ruta_str = self._validar_ruta(ruta)
        archivos = self._zip_leer_archivos(ruta_str)

        # Shared strings
        shared = []
        if 'xl/sharedStrings.xml' in archivos:
            shared = self._xlsx_shared_strings(archivos['xl/sharedStrings.xml'])

        # Workbook + relaciones
        if 'xl/workbook.xml' not in archivos:
            raise RuntimeError(f"Archivo xlsx inválido (falta workbook.xml): {ruta_str}")
        if 'xl/_rels/workbook.xml.rels' not in archivos:
            raise RuntimeError(f"Archivo xlsx inválido (falta workbook.xml.rels): {ruta_str}")

        hojas_info = self._xlsx_workbook_sheets(
            archivos['xl/workbook.xml'],
            archivos['xl/_rels/workbook.xml.rels']
        )

        sheets = {}
        for nombre, ruta_hoja in hojas_info:
            if ruta_hoja in archivos:
                sheets[nombre] = archivos[ruta_hoja]

        if not sheets:
            raise RuntimeError(f"No se encontraron hojas en: {ruta_str}")

        return sheets, shared

    def leerxlsx(self, ruta):
        """Lee la primera hoja completa (con encabezado)."""
        sheets, shared = self._xlsx_abrir(ruta)
        primera = next(iter(sheets.values()))
        return self._xlsx_parsear_hoja(primera, shared)

    def leerxlsx_datos(self, ruta):
        """Lee la primera hoja omitiendo la primera fila (encabezado)."""
        tabla = self.leerxlsx(ruta)
        return tabla[1:] if len(tabla) > 1 else []

    def leerxlsx_hoja(self, ruta, nombre_hoja):
        """Lee una hoja específica por nombre."""
        sheets, shared = self._xlsx_abrir(ruta)
        nombre_hoja = str(nombre_hoja)
        if nombre_hoja not in sheets:
            disponibles = ", ".join(sheets.keys())
            raise RuntimeError(
                f"La hoja '{nombre_hoja}' no existe en '{ruta}'. "
                f"Disponibles: {disponibles}"
            )
        return self._xlsx_parsear_hoja(sheets[nombre_hoja], shared)

    def leerxlsx_columna(self, ruta, col):
        """Lee una columna (índice 0) de la primera hoja, sin encabezado."""
        col   = int(col)
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
        """Retorna los nombres de todas las hojas del archivo."""
        sheets, _ = self._xlsx_abrir(ruta)
        return list(sheets.keys())

    # ═══════════════════════════════════════════════════════════
    # HELPERS CSV
    # ═══════════════════════════════════════════════════════════

    def _parsear_fila_csv(self, linea):
        celdas      = []
        celda       = []
        en_comillas = False
        i = 0
        while i < len(linea):
            c = linea[i]
            if c == '"':
                if en_comillas and i+1 < len(linea) and linea[i+1] == '"':
                    celda.append('"')
                    i += 2
                    continue
                en_comillas = not en_comillas
            elif c == ',' and not en_comillas:
                celdas.append(self._convertir_celda(''.join(celda).strip()))
                celda = []
            else:
                celda.append(c)
            i += 1
        celdas.append(self._convertir_celda(''.join(celda).strip()))
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

    # ═══════════════════════════════════════════════════════════
    # VALIDADORES
    # ═══════════════════════════════════════════════════════════

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
