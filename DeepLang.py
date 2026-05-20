#!/usr/bin/env python3
# DeepLang.py — Punto de entrada principal del intérprete DeepLang

import sys
import os

from lexer      import DeepLangLexer,  LexerError
from parser     import DeepLangParser, ParseError
from EvalVisitor import EvalVisitor


# ─────────────────────────────────────────────────────────────
# Contar bloques abiertos (para el REPL multilínea)
# ─────────────────────────────────────────────────────────────
def contar_bloques_abiertos(texto):
    try:
        tokens = DeepLangLexer(texto + '\n').nextToken()
    except LexerError:
        return 0
    from lexer import IF, WHILE, FUN, END
    profundidad = 0
    for tok in tokens:
        if tok.type in (IF, WHILE, FUN):
            profundidad += 1
        elif tok.type == END:
            profundidad -= 1
    return max(profundidad, 0)


# ─────────────────────────────────────────────────────────────
# Ejecutar un bloque de código con un visitor dado
# ─────────────────────────────────────────────────────────────
def ejecutar(codigo, eval_, nombre="<stdin>"):
    try:
        tokens = DeepLangLexer(codigo + '\n').nextToken()
        tree   = DeepLangParser(tokens).prog()
        eval_.visit(tree)
        return True
    except LexerError    as e: print(f"[Error léxico]      {e}")
    except ParseError    as e: print(f"[Error sintáctico]  {e}")
    except ZeroDivisionError as e: print(f"[Error aritmético]  {e}")
    except Exception     as e: print(f"[Error] {e}")
    return False


# ─────────────────────────────────────────────────────────────
# Cargar un archivo .dl o cualquier script DeepLang
# ─────────────────────────────────────────────────────────────
def cargar_archivo(ruta, eval_):
    if not os.path.isfile(ruta):
        print(f"[Error] No se encontró el archivo: {ruta}")
        return False
    with open(ruta, encoding='utf-8') as f:
        codigo = f.read()
    return ejecutar(codigo, eval_, nombre=ruta)


# ─────────────────────────────────────────────────────────────
# REPL interactivo
# ─────────────────────────────────────────────────────────────
def repl(eval_=None):
    if eval_ is None:
        eval_ = EvalVisitor()

    print("DeepLang | escribe 'salir' para terminar")
    print("  Comandos especiales:")
    print("    cargar \"archivo.dl\"   — carga un módulo DeepLang")
    print("    salir                  — termina el REPL")
    print("─" * 60)

    buffer = []
    while True:
        prompt = '... ' if buffer else 'dl> '
        try:
            line = input(prompt)
        except (EOFError, KeyboardInterrupt):
            print("\nHasta luego.")
            break

        stripped = line.strip().lower()
        if stripped in ('salir', 'exit', 'quit'):
            print("Hasta luego.")
            break

        # Comando especial: cargar "modulo.dl"
        if stripped.startswith('cargar '):
            partes = line.strip()[7:].strip()
            # Quitar comillas si las hay
            nombre = partes.strip('"').strip("'")
            cargar_archivo(nombre, eval_)
            buffer = []
            continue

        buffer.append(line)
        codigo = '\n'.join(buffer)

        if contar_bloques_abiertos(codigo) > 0:
            continue

        if codigo.strip() == '':
            buffer = []
            continue

        ejecutar(codigo, eval_)
        buffer = []


# ─────────────────────────────────────────────────────────────
# Punto de entrada
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    args = sys.argv[1:]

    if args:
        eval_ = EvalVisitor()
        # Ejecutar todos los archivos pasados en orden
        for archivo in args:
            if not os.path.isfile(archivo):
                print(f"[Error] No se encontró: {archivo}")
                sys.exit(1)
            print(f"─── Ejecutando: {archivo} {'─'*30}")
            with open(archivo, encoding='utf-8') as f:
                codigo = f.read()
            ejecutar(codigo, eval_, nombre=archivo)
        print("─" * 60)
        # Tras ejecutar los archivos, abrir REPL con el estado cargado
        # (útil para explorar después de cargar módulos)
        if len(args) == 1 and args[0].endswith('.dl'):
            repl(eval_)
    else:
        repl()
