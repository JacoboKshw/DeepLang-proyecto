#!/usr/bin/env python3
from lexer            import DeepLangLexer, LexerError
from parser           import DeepLangParser, ParseError
from EvalVisitor      import EvalVisitor
from deeplang_filelib import DeepLangFileLib

_filelib = DeepLangFileLib()

def _argv():
    try:
        # /proc/self/cmdline separa los argumentos con bytes nulos
        contenido = _filelib.leerarchivo('/proc/self/cmdline')
        partes    = [p for p in contenido.split('\x00') if p]
        # partes[0]=python3  partes[1]=DeepLang.py  partes[2:]=args del usuario
        return partes[2:] if len(partes) > 2 else []
    except Exception:
        return []


def _salir(codigo=0):
    raise SystemExit(codigo)

# Contar bloques abiertos (para el REPL multilínea)
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


# Ejecutar un bloque de código con un visitor dado

def ejecutar(codigo, eval_, nombre="<stdin>"):
    try:
        tokens = DeepLangLexer(codigo + '\n').nextToken()
        tree   = DeepLangParser(tokens).prog()
        eval_.visit(tree)
        return True
    except LexerError        as e: print(f"[Error léxico]      {e}")
    except ParseError        as e: print(f"[Error sintáctico]  {e}")
    except ZeroDivisionError as e: print(f"[Error aritmético]  {e}")
    except Exception         as e: print(f"[Error] {e}")
    return False

# Cargar un archivo .dl usando DeepLangFileLib
def cargar_archivo(ruta, eval_):
    try:
        codigo = _filelib.leerarchivo(ruta)
    except RuntimeError as e:
        print(f"[Error] {e}")
        return False
    return ejecutar(codigo, eval_, nombre=ruta)


def repl(eval_=None):
    if eval_ is None:
        eval_ = EvalVisitor()

    print("DeepLang | escribe 'salir' para terminar")
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
            nombre = line.strip()[7:].strip().strip('"').strip("'")
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


# Punto de entrada

if __name__ == '__main__':
    args = _argv()

    if args:
        eval_ = EvalVisitor()
        for archivo in args:
            try:
                codigo = _filelib.leerarchivo(archivo)
            except RuntimeError as e:
                print(f"[Error] {e}")
                _salir(1)
            print(f"─── Ejecutando: {archivo} {'─'*30}")
            ejecutar(codigo, eval_, nombre=archivo)
        print("─" * 60)
        # Tras ejecutar los archivos, abrir REPL con el estado cargado
        if len(args) == 1 and args[0].endswith('.dl'):
            repl(eval_)
    else:
        repl()
