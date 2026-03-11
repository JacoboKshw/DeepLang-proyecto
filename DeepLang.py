#!/usr/bin/env python3
# DeepLang.py

import sys

from lexer       import DeepLangLexer,  LexerError
from parser      import DeepLangParser, ParseError
from EvalVisitor import EvalVisitor


def main():
    # String inputFile = null;
    # if (args.length > 0) inputFile = args[0];
    input_file = sys.argv[1] if len(sys.argv) > 1 else None

    if input_file:
        # InputStream is = new FileInputStream(inputFile);
        with open(input_file, encoding='utf-8') as f:
            source = f.read()
        run(source)
    else:
        # Lee desde teclado igual que System.in
        repl()


def run(source):
    try:
        # ANTLRInputStream input 
        # DeepLangLexer lexer
        lexer = DeepLangLexer(source)

        # CommonTokenStream tokens
        tokens = lexer.nextToken()

        # DeepLangParser parser = new DeepLangParser(tokens);
        parser = DeepLangParser(tokens)

        # ParseTree tree 
        tree = parser.prog()

        # EvalVisitor eval
        eval = EvalVisitor()

        # eval.visit(tree);
        eval.visit(tree)

    except LexerError        as e: print(f"[Error léxico]     {e}")
    except ParseError        as e: print(f"[Error sintáctico] {e}")
    except ZeroDivisionError as e: print(f"[Error aritmético] {e}")
    except Exception         as e: print(f"[Error]            {e}")


def repl():
    """Modo interactivo — equivalente a leer de System.in línea por línea."""
    print("DeepLang  |  + - * /  |  escribe 'salir' para terminar")
    print("─" * 48)
    # En modo REPL el visitor persiste para recordar variables
    eval_ = EvalVisitor()

    while True:
        try:
            line = input("dl> ")
        except (EOFError, KeyboardInterrupt):
            print("\nHasta luego.")
            break

        if line.strip().lower() in ('salir', 'exit', 'quit'):
            print("Hasta luego.")
            break

        if line.strip() == '':
            continue

        try:
            lexer  = DeepLangLexer(line + '\n')
            tokens = lexer.nextToken()
            parser = DeepLangParser(tokens)
            tree   = parser.prog()
            eval_.visit(tree)
        except LexerError        as e: print(f"[Error léxico]     {e}")
        except ParseError        as e: print(f"[Error sintáctico] {e}")
        except ZeroDivisionError as e: print(f"[Error aritmético] {e}")
        except Exception         as e: print(f"[Error]            {e}")


if __name__ == '__main__':
    main()
