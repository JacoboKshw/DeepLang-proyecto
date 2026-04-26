#!/usr/bin/env python3
# DeepLang.py

from lexer       import DeepLangLexer,  LexerError
from parser      import DeepLangParser, ParseError
from EvalVisitor import EvalVisitor


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


def repl():
    print("DeepLang  |  escribe 'salir' para terminar")
    print("─" * 60)
    eval_ = EvalVisitor()
    buffer = []

    while True:
        prompt = '... ' if buffer else 'dl> '

        try:
            line = input(prompt)
        except (EOFError, KeyboardInterrupt):
            print("\nHasta luego.")
            break

        if line.strip().lower() in ('salir', 'exit', 'quit'):
            print("Hasta luego.")
            break

        buffer.append(line)
        codigo = '\n'.join(buffer)

        if contar_bloques_abiertos(codigo) > 0:
            continue

        if codigo.strip() == '':
            buffer = []
            continue

        try:
            lexer  = DeepLangLexer(codigo + '\n')
            tokens = lexer.nextToken()
            parser = DeepLangParser(tokens)
            tree   = parser.prog()
            eval_.visit(tree)
        except LexerError        as e: print(f"[Error léxico]     {e}")
        except ParseError        as e: print(f"[Error sintáctico] {e}")
        except ZeroDivisionError as e: print(f"[Error aritmético] {e}")
        except Exception         as e: print(f"[Error]            {e}")

        buffer = []


if __name__ == '__main__':
    repl()
