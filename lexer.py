# lexer.py
# Tokens 

MUL     = 'MUL'
DIV     = 'DIV'
ADD     = 'ADD'
SUB     = 'SUB'
ID      = 'ID'
INT     = 'INT'
NEWLINE = 'NEWLINE'
LPAREN  = 'LPAREN'
RPAREN  = 'RPAREN'
EOF     = 'EOF'

# ── Tokens para condicionales y ciclos ───────────────────
IF       = 'IF'
ELSE     = 'ELSE'
END      = 'END'
WHILE    = 'WHILE'
EQ       = 'EQ'       # ==
NEQ      = 'NEQ'      # !=
LT       = 'LT'       # <
GT       = 'GT'       # >
LTE      = 'LTE'      # <=
GTE      = 'GTE'      # >=

# ── Tokens para arreglos ──────────────────────────────────
LBRACKET = 'LBRACKET'  # [
RBRACKET = 'RBRACKET'  # ]
COMMA    = 'COMMA'     # ,
POW      = 'POW'       # ^

# ── Tokens para I/O ───────────────────────────────────────
PRINT    = 'PRINT'

# ── Tokens para funciones ─────────────────────────────────
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


# Palabras reservadas → tipo de token
KEYWORDS = {
    'if':     IF,
    'else':   ELSE,
    'end':    END,
    'while':  WHILE,
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

            # WS -> skip  
            if ch in (' ', '\t'):
                self.advance()
                continue

            # NEWLINE : '\r'? '\n'
            if ch in ('\r', '\n'):
                while self.current() in ('\r', '\n'):
                    self.advance()
                tokens.append(Token(NEWLINE, '\n', self.line))
                continue

            # INT : [0-9]+
            if ch.isdigit():
                start = self.pos
                line  = self.line
                while self.current() and self.current().isdigit():
                    self.advance()
                tokens.append(Token(INT, self.src[start:self.pos], line))
                continue

            # ID / palabras reservadas : [a-zA-Z]+
            if ch.isalpha():
                start = self.pos
                line  = self.line
                while self.current() and self.current().isalpha():
                    self.advance()
                word = self.src[start:self.pos]
                # ¿es palabra reservada?
                tok_type = KEYWORDS.get(word, ID)
                tokens.append(Token(tok_type, word, line))
                continue

            # ── Operadores de dos caracteres primero ──────
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
