# parser.py
from lexer import (
    MUL, DIV, ADD, SUB, POW, ID, INT, FLOAT, STRING, NEWLINE, LPAREN, RPAREN, EOF,
    IF, ELSE, END, WHILE, FOR, TO,
    EQ, NEQ, LT, GT, LTE, GTE,
    LBRACKET, RBRACKET, COMMA,
    PRINT, FUN, RETURN,
)


class ParseError(Exception):
    pass


# ──────────────────────────────────────────────
# Nodos del árbol sintáctico
# ──────────────────────────────────────────────

class ProgContext:
    """prog: stat+"""
    def __init__(self, stats):
        self.stats = stats

class AssignContext:
    def __init__(self, name, expr):
        self.name = name
        self.expr = expr

class PrintExprContext:
    """expr sola en una línea — evalúa pero no imprime"""
    def __init__(self, expr):
        self.expr = expr

class PrintContext:
    """print(expr)"""
    def __init__(self, expr):
        self.expr = expr

class MulDivContext:
    def __init__(self, left, op, right):
        self.left  = left
        self.op    = op
        self.right = right

class PowContext:
    def __init__(self, left, op, right):
        self.left  = left
        self.op    = op
        self.right = right

class AddSubContext:
    def __init__(self, left, op, right):
        self.left  = left
        self.op    = op
        self.right = right

class UnaryMinusContext:
    def __init__(self, expr):
        self.expr = expr

class IntContext:
    def __init__(self, value):
        self.value = value

class FloatContext:          # <-- NUEVO
    def __init__(self, value):
        self.value = value

class StringContext:
    def __init__(self, value):
        self.value = value

class IdContext:
    def __init__(self, name):
        self.name = name

class ParensContext:
    def __init__(self, expr):
        self.expr = expr

class ConditionContext:
    def __init__(self, left, op, right):
        self.left  = left
        self.op    = op
        self.right = right

class IfContext:
    def __init__(self, cond, then_stats, else_stats=None):
        self.cond       = cond
        self.then_stats = then_stats
        self.else_stats = else_stats or []

class WhileContext:
    def __init__(self, cond, stats):
        self.cond  = cond
        self.stats = stats

class ForContext:
    def __init__(self, var, start, end_, body):
        self.var   = var
        self.start = start
        self.end_  = end_
        self.body  = body

class ArrayLiteralContext:
    def __init__(self, elements):
        self.elements = elements

class ArrayAccessContext:
    def __init__(self, name, index):
        self.name  = name
        self.index = index

class ArrayAssignContext:
    def __init__(self, name, index, expr):
        self.name  = name
        self.index = index
        self.expr  = expr

class FuncDefContext:
    def __init__(self, name, params, body):
        self.name   = name
        self.params = params
        self.body   = body

class FuncCallContext:
    def __init__(self, name, args):
        self.name = name
        self.args = args

class ReturnContext:
    def __init__(self, expr):
        self.expr = expr


# ──────────────────────────────────────────────
# Parser principal
# ──────────────────────────────────────────────

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

    def prog(self):
        self.skip_nl()
        stats = []
        while not self.match(EOF):
            s = self.stat()
            if s is not None:
                stats.append(s)
            self.skip_nl()
        return ProgContext(stats)

    def stat(self):
        if self.match(NEWLINE):
            self.consume()
            return None

        if self.match(IF):
            return self.if_stat()

        if self.match(WHILE):
            return self.while_stat()

        if self.match(FOR):
            return self.for_stat()

        if self.match(PRINT):
            return self.print_stat()

        if self.match(FUN):
            return self.fun_def()

        if self.match(RETURN):
            return self.return_stat()

        # Asignaciones simples y de arreglo
        if self.match(ID) and self.pos + 1 < len(self.tokens):
            next_tok = self.tokens[self.pos + 1]

            if next_tok.type == LBRACKET:
                depth = 1
                look  = self.pos + 2
                while look < len(self.tokens) and depth > 0:
                    t = self.tokens[look]
                    if t.type == LBRACKET: depth += 1
                    if t.type == RBRACKET: depth -= 1
                    look += 1
                is_assign = (look < len(self.tokens) and
                             self.tokens[look].text == '=')
                if is_assign:
                    name = self.consume(ID).text
                    self.consume(LBRACKET)
                    index = self.expr()
                    self.consume(RBRACKET)
                    self.consume()   # consume '='
                    expr = self.expr()
                    if self.match(NEWLINE): self.consume()
                    return ArrayAssignContext(name, index, expr)

            if next_tok.text == '=':
                name = self.consume(ID).text
                self.consume()       # consume '='
                expr = self.expr()
                if self.match(NEWLINE): self.consume()
                return AssignContext(name, expr)

        e = self.expr()
        if self.match(NEWLINE): self.consume()
        return PrintExprContext(e)

    def print_stat(self):
        self.consume(PRINT)
        self.consume(LPAREN)
        e = self.expr()
        self.consume(RPAREN)
        if self.match(NEWLINE): self.consume()
        return PrintContext(e)

    def fun_def(self):
        self.consume(FUN)
        name = self.consume(ID).text
        self.consume(LPAREN)
        params = []
        if not self.match(RPAREN):
            params.append(self.consume(ID).text)
            while self.match(COMMA):
                self.consume()
                params.append(self.consume(ID).text)
        self.consume(RPAREN)
        if self.match(NEWLINE): self.consume()
        body = []
        self.skip_nl()
        while not self.match(END) and not self.match(EOF):
            s = self.stat()
            if s is not None:
                body.append(s)
            self.skip_nl()
        self.consume(END)
        if self.match(NEWLINE): self.consume()
        return FuncDefContext(name, params, body)

    def return_stat(self):
        self.consume(RETURN)
        e = self.expr()
        if self.match(NEWLINE): self.consume()
        return ReturnContext(e)

    def if_stat(self):
        self.consume(IF)
        cond = self.condition()
        if self.match(NEWLINE): self.consume()
        then_stats = []
        self.skip_nl()
        while not self.match(ELSE) and not self.match(END) and not self.match(EOF):
            s = self.stat()
            if s is not None:
                then_stats.append(s)
            self.skip_nl()
        else_stats = []
        if self.match(ELSE):
            self.consume(ELSE)
            if self.match(NEWLINE): self.consume()
            self.skip_nl()
            while not self.match(END) and not self.match(EOF):
                s = self.stat()
                if s is not None:
                    else_stats.append(s)
                self.skip_nl()
        self.consume(END)
        if self.match(NEWLINE): self.consume()
        return IfContext(cond, then_stats, else_stats)

    def while_stat(self):
        self.consume(WHILE)
        cond = self.condition()
        if self.match(NEWLINE): self.consume()
        body = []
        self.skip_nl()
        while not self.match(END) and not self.match(EOF):
            s = self.stat()
            if s is not None:
                body.append(s)
            self.skip_nl()
        self.consume(END)
        if self.match(NEWLINE): self.consume()
        return WhileContext(cond, body)

    def for_stat(self):
        self.consume(FOR)
        var = self.consume(ID).text
        self.consume()   # consume '='
        start = self.expr()
        self.consume(TO)
        end_ = self.expr()
        if self.match(NEWLINE): self.consume()
        body = []
        self.skip_nl()
        while not self.match(END) and not self.match(EOF):
            s = self.stat()
            if s is not None:
                body.append(s)
            self.skip_nl()
        self.consume(END)
        if self.match(NEWLINE): self.consume()
        return ForContext(var, start, end_, body)

    def condition(self):
        left = self.expr()
        if not self.match(EQ, NEQ, LT, GT, LTE, GTE):
            raise ParseError(
                f"Línea {self.current().line}: "
                f"se esperaba operador de comparación (==, !=, <, >, <=, >=), "
                f"encontré '{self.current().type}'"
            )
        op    = self.consume()
        right = self.expr()
        return ConditionContext(left, op, right)

    def expr(self):
        return self._add()

    def _add(self):
        left = self._mul()
        while self.match(ADD, SUB):
            op    = self.consume()
            right = self._mul()
            left  = AddSubContext(left, op, right)
        return left

    def _mul(self):
        left = self._pow()
        while self.match(MUL, DIV):
            op    = self.consume()
            right = self._pow()
            left  = MulDivContext(left, op, right)
        return left

    def _pow(self):
        left = self._primary()
        if self.match(POW):
            op    = self.consume()
            right = self._pow()   # asociatividad derecha
            return PowContext(left, op, right)
        return left

    def _primary(self):
        tok = self.current()

        if tok.type == SUB:
            self.consume()
            return UnaryMinusContext(self._primary())

        if tok.type == LBRACKET:
            self.consume()
            self.skip_nl()
            elements = []
            if not self.match(RBRACKET):
                elements.append(self.expr())
                self.skip_nl()
                while self.match(COMMA):
                    self.consume()
                    self.skip_nl()
                    elements.append(self.expr())
                    self.skip_nl()
            self.consume(RBRACKET)
            return ArrayLiteralContext(elements)

        if tok.type == INT:
            self.consume()
            return IntContext(int(tok.text))

        if tok.type == FLOAT:          # <-- NUEVO
            self.consume()
            return FloatContext(float(tok.text))

        if tok.type == STRING:
            self.consume()
            return StringContext(tok.text)

        if tok.type == LPAREN:
            self.consume()
            e = self.expr()
            self.consume(RPAREN)
            return ParensContext(e)

        if tok.type == ID:
            name = self.consume().text
            if self.match(LPAREN):
                self.consume()
                args = []
                if not self.match(RPAREN):
                    args.append(self.expr())
                    while self.match(COMMA):
                        self.consume()
                        args.append(self.expr())
                self.consume(RPAREN)
                return FuncCallContext(name, args)
            if self.match(LBRACKET):
                self.consume()
                index = self.expr()
                self.consume(RBRACKET)
                return ArrayAccessContext(name, index)
            return IdContext(name)

        raise ParseError(
            f"Línea {tok.line}: token inesperado '{tok.type}' ({tok.text!r})"
        )
