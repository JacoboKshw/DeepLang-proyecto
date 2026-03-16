# parser.py

# Produce el mismo árbol

from lexer import MUL, DIV, ADD, SUB, ID, INT, NEWLINE, LPAREN, RPAREN, EOF


class ParseError(Exception):
    pass


# ── Nodos del árbol

class ProgContext:
    """prog: stat+"""
    def __init__(self, stats):
        self.stats = stats


class AssignContext:
    """ # assign"""
    def __init__(self, name, expr):
        self.name = name    # equivalente a ctx.ID().getText()
        self.expr = expr    # equivalente a ctx.expr()


class PrintExprContext:
    """ # printExpr"""
    def __init__(self, expr):
        self.expr = expr    # equivalente a ctx.expr()


class MulDivContext:
    """# MulDiv"""
    def __init__(self, left, op, right):
        self.left  = left
        self.op    = op     # equivalente a ctx.op.getType()
        self.right = right


class AddSubContext:
    """# AddSub"""
    def __init__(self, left, op, right):
        self.left  = left
        self.op    = op     # equivalente a ctx.op.getType()
        self.right = right


class IntContext:
    """# int"""
    def __init__(self, value):
        self.value = value  # equivalente a ctx.INT().getText()


class IdContext:
    """# id"""
    def __init__(self, name):
        self.name = name    # equivalente a ctx.ID().getText()


class ParensContext:
    """# parens"""
    def __init__(self, expr):
        self.expr = expr    # equivalente a ctx.expr()


# ── Parser ─────────────────────────────────────────────────

class DeepLangParser:

    def __init__(self, tokens):
        self.tokens = tokens
        self.pos    = 0

    def current(self):
        return self.tokens[self.pos]

    def match(self, *types):
        return self.current().type in types

    def consume(self, type_=None):
        tok = self.current()
        if type_ and tok.type != type_:
            raise ParseError(
                f"Línea {tok.line}: se esperaba '{type_}', "
                f"encontré '{tok.type}' ({tok.text!r})"
            )
        self.pos += 1
        return tok

    def skip_nl(self):
        while self.match(NEWLINE):
            self.consume()

    # ── prog: stat+ ────────────────────────────────────────
    def prog(self):
        """Equivalente a parser.prog() en Calc.java"""
        self.skip_nl()
        stats = []
        while not self.match(EOF):
            s = self.stat()
            if s is not None:
                stats.append(s)
            self.skip_nl()
        return ProgContext(stats)

    # ── stat ───────────────────────────────────────────────
    def stat(self):
        # blank
        if self.match(NEWLINE):
            self.consume()
            return None

        # assign:  ID '=' expr NEWLINE
        if self.match(ID) and self.pos + 1 < len(self.tokens):
            next_tok = self.tokens[self.pos + 1]
            if next_tok.text == '=':
                name = self.consume(ID).text
                self.consume()          # consume '='
                expr = self.expr()
                if self.match(NEWLINE):
                    self.consume()
                return AssignContext(name, expr)

        # expr NEWLINE
        e = self.expr()
        if self.match(NEWLINE):
            self.consume()
        return PrintExprContext(e)

    # ── expr ───────────────────────────────────────────────
    def expr(self):
        return self._add()

    # MulDiv tiene mayor precedencia que AddSub
    def _add(self):
        left = self._mul()
        while self.match(ADD, SUB):
            op    = self.consume()      
            right = self._mul()
            left  = AddSubContext(left, op, right)
        return left

    def _mul(self):
        left = self._primary()
        while self.match(MUL, DIV):
            op    = self.consume()      
            right = self._primary()
            left  = MulDivContext(left, op, right)
        return left

    def _primary(self):
        tok = self.current()

        # INT
        if tok.type == INT:
            self.consume()
            return IntContext(int(tok.text))

        # '(' expr ')'
        if tok.type == LPAREN:
            self.consume()
            e = self.expr()
            self.consume(RPAREN)
            return ParensContext(e)

        # ID
        if tok.type == ID:
            self.consume()
            return IdContext(tok.text)

        raise ParseError(
            f"Línea {tok.line}: token inesperado '{tok.type}' ({tok.text!r})"
        )
