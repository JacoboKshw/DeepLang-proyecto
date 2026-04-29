# lexer.py
# Tokens básicos del lenguaje.

MUL     = 'MUL'
DIV     = 'DIV'
ADD     = 'ADD'
SUB     = 'SUB'
ID      = 'ID'
INT     = 'INT'
STRING  = 'STRING'
NEWLINE = 'NEWLINE'
LPAREN  = 'LPAREN'
RPAREN  = 'RPAREN'
EOF     = 'EOF'

# Tokens para estructuras de control.
IF       = 'IF'
ELSE     = 'ELSE'
END      = 'END'
WHILE    = 'WHILE'
FOR      = 'FOR'
TO       = 'TO'
EQ       = 'EQ'       # ==
NEQ      = 'NEQ'      # !=
LT       = 'LT'       # <
GT       = 'GT'       # >
LTE      = 'LTE'      # <=
GTE      = 'GTE'      # >=

# Tokens para arreglos.
LBRACKET = 'LBRACKET'  # [
RBRACKET = 'RBRACKET'  # ]
COMMA    = 'COMMA'     # ,
POW      = 'POW'       # ^

# Tokens de entrada/salida.
PRINT    = 'PRINT'

# Tokens para funciones.
FUN      = 'FUN'
RETURN   = 'RETURN'


class Token:
    def __init__(self, type_, text, line):
        self.type = type_
        self.text = text
        self.line = line

    def __repr__(self):
        return f'Token({self.type}, {self.text!r})'


class LexerError(Exception):
    pass


# Palabras reservadas y su tipo de token.
KEYWORDS = {
    'if':     IF,
    'else':   ELSE,
    'end':    END,
    'while':  WHILE,
    'for':    FOR,
    'to':     TO,
    'print':  PRINT,
    'fun':    FUN,
    'return': RETURN,
}


class DeepLangLexer:

    def __init__(self, input_stream):
        self.src  = input_stream
        self.pos  = 0
        self.line = 1

    def current(self):
        return self.src[self.pos] if self.pos < len(self.src) else None

    def advance(self):
        ch = self.src[self.pos]
        self.pos += 1
        if ch == '\n':
            self.line += 1
        return ch

    def nextToken(self):
        tokens = []

        while self.pos < len(self.src):
            ch = self.current()

            # Ignoramos espacios y tabulaciones.
            if ch in (' ', '\t'):
                self.advance()
                continue

            # Normalizamos saltos de línea como NEWLINE.
            if ch in ('\r', '\n'):
                while self.current() in ('\r', '\n'):
                    self.advance()
                tokens.append(Token(NEWLINE, '\n', self.line))
                continue

            # Enteros.
            if ch.isdigit():
                start = self.pos
                line  = self.line
                while self.current() and self.current().isdigit():
                    self.advance()
                tokens.append(Token(INT, self.src[start:self.pos], line))
                continue

            # Cadenas entre comillas dobles.
            if ch == '"':
                line = self.line
                self.advance()
                chars = []
                while self.current() is not None and self.current() != '"':
                    if self.current() == '\\':
                        self.advance()
                        esc = self.current()
                        if esc is None:
                            raise LexerError(f"Cadena sin cerrar en línea {line}")
                        escapes = {'n': '\n', 't': '\t', '"': '"', '\\': '\\'}
                        chars.append(escapes.get(esc, esc))
                        self.advance()
                    else:
                        chars.append(self.advance())
                if self.current() != '"':
                    raise LexerError(f"Cadena sin cerrar en línea {line}")
                self.advance()
                tokens.append(Token(STRING, ''.join(chars), line))
                continue

            # Identificadores o palabras reservadas.
            if ch.isalpha():
                start = self.pos
                line  = self.line
                while self.current() and self.current().isalpha():
                    self.advance()
                word = self.src[start:self.pos]
                tok_type = KEYWORDS.get(word, ID)
                tokens.append(Token(tok_type, word, line))
                continue

            # Revisamos primero operadores de dos caracteres.
            two = self.src[self.pos:self.pos+2]
            if two == '==':
                tokens.append(Token(EQ,  '==', self.line)); self.pos += 2; continue
            if two == '!=':
                tokens.append(Token(NEQ, '!=', self.line)); self.pos += 2; continue
            if two == '<=':
                tokens.append(Token(LTE, '<=', self.line)); self.pos += 2; continue
            if two == '>=':
                tokens.append(Token(GTE, '>=', self.line)); self.pos += 2; continue

            simple = {
                '*': MUL,
                '/': DIV,
                '+': ADD,
                '-': SUB,
                '^': POW,
                '(': LPAREN,
                ')': RPAREN,
                '=': 'ASSIGN',
                '<': LT,
                '>': GT,
                '[': LBRACKET,
                ']': RBRACKET,
                ',': COMMA,
            }
            if ch in simple:
                tokens.append(Token(simple[ch], ch, self.line))
                self.advance()
                continue

            raise LexerError(f"Carácter desconocido '{ch}' en línea {self.line}")

        tokens.append(Token(EOF, None, self.line))
        return tokens
