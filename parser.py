# parser.py

from lexer import (
    MUL, DIV, ADD, SUB, POW, ID, INT, STRING, NEWLINE, LPAREN, RPAREN, EOF,
    IF, ELSE, END, WHILE, FOR, TO,
    EQ, NEQ, LT, GT, LTE, GTE,
    LBRACKET, RBRACKET, COMMA,
    PRINT, FUN, RETURN,
)


class ParseError(Exception):
    pass


# Nodos del árbol sintáctico

class ProgContext:
    """prog: stat+"""
    def __init__(self, stats):
        self.stats = stats


class AssignContext:
    """Asignación simple."""
    def __init__(self, name, expr):
        self.name = name
        self.expr = expr


class PrintExprContext:
    """expr sola en una línea — ya NO imprime, solo evalúa"""
    def __init__(self, expr):
        self.expr = expr


class PrintContext:
    """print(expr) — imprime el valor"""
    def __init__(self, expr):
        self.expr = expr


class MulDivContext:
    """Multiplicación o división."""
    def __init__(self, left, op, right):
        self.left  = left
        self.op    = op
        self.right = right


class PowContext:
    """Potencia."""
    def __init__(self, left, op, right):
        self.left  = left
        self.op    = op
        self.right = right


class AddSubContext:
    """Suma o resta."""
    def __init__(self, left, op, right):
        self.left  = left
        self.op    = op
        self.right = right


class UnaryMinusContext:
    """Negación unaria: -expr."""
    def __init__(self, expr):
        self.expr = expr


class IntContext:
    """Literal entero."""
    def __init__(self, value):
        self.value = value


class StringContext:
    """Literal de texto."""
    def __init__(self, value):
        self.value = value


class IdContext:
    """Uso de variable."""
    def __init__(self, name):
        self.name = name


class ParensContext:
    """Expresión entre paréntesis."""
    def __init__(self, expr):
        self.expr = expr


class ConditionContext:
    """
    Una condición: expr op expr
    op puede ser ==, !=, <, >, <=, >=
    """
    def __init__(self, left, op, right):
        self.left  = left
        self.op    = op
        self.right = right


class IfContext:
    """
    if cond
        stats_then
    [else
        stats_else]
    end
    """
    def __init__(self, cond, then_stats, else_stats=None):
        self.cond       = cond
        self.then_stats = then_stats
        self.else_stats = else_stats or []


class WhileContext:
    """
    while cond
        stats
    end
    """
    def __init__(self, cond, stats):
        self.cond  = cond
        self.stats = stats


class ForContext:
    """
    for var = inicio to fin
        stats
    end
    """
    def __init__(self, var, start, end_, body):
        self.var   = var    # str
        self.start = start  # expr
        self.end_  = end_   # expr
        self.body  = body   # lista de stat


class ArrayLiteralContext:
    """lista = [1, 2, 3]"""
    def __init__(self, elements):
        self.elements = elements


class ArrayAccessContext:
    """lista[0]  — como expresión"""
    def __init__(self, name, index):
        self.name  = name
        self.index = index


class ArrayAssignContext:
    """lista[0] = 5"""
    def __init__(self, name, index, expr):
        self.name  = name
        self.index = index
        self.expr  = expr


class FuncDefContext:
    """
    fun nombre(a, b)
        stats
    end
    """
    def __init__(self, name, params, body):
        self.name   = name
        self.params = params
        self.body   = body


class FuncCallContext:
    """nombre(expr, expr)  — como expresión"""
    def __init__(self, name, args):
        self.name = name
        self.args = args


class ReturnContext:
    """return expr"""
    def __init__(self, expr):
        self.expr = expr


# Parser principal de DeepLang

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

    # Parsea el programa completo: una o más sentencias.
    def prog(self):
        self.skip_nl()
        stats = []
        while not self.match(EOF):
            s = self.stat()
            if s is not None:
                stats.append(s)
            self.skip_nl()
        return ProgContext(stats)

    # Parsea una sentencia.
    def stat(self):
        # Línea en blanco.
        if self.match(NEWLINE):
            self.consume()
            return None

        # Bloque if.
        if self.match(IF):
            return self.if_stat()

        # Bloque while.
        if self.match(WHILE):
            return self.while_stat()

        # Bloque for.
        if self.match(FOR):
            return self.for_stat()

        # Llamada a print.
        if self.match(PRINT):
            return self.print_stat()

        # Definición de función.
        if self.match(FUN):
            return self.fun_def()

        # Return dentro de función.
        if self.match(RETURN):
            return self.return_stat()

        # Asignaciones simples y de arreglo.
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
                    self.consume()
                    expr = self.expr()
                    if self.match(NEWLINE):
                        self.consume()
                    return ArrayAssignContext(name, index, expr)

            if next_tok.text == '=':
                name = self.consume(ID).text
                self.consume()
                expr = self.expr()
                if self.match(NEWLINE):
                    self.consume()
                return AssignContext(name, expr)

        # Si no coincide nada anterior, se trata como expresión.
        e = self.expr()
        if self.match(NEWLINE):
            self.consume()
        return PrintExprContext(e)

    # Parsea una sentencia print(expr).
    def print_stat(self):
        self.consume(PRINT)
        self.consume(LPAREN)
        e = self.expr()
        self.consume(RPAREN)
        if self.match(NEWLINE):
            self.consume()
        return PrintContext(e)

    # Parsea definición de función: fun nombre(...) ... end.
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
        if self.match(NEWLINE):
            self.consume()

        body = []
        self.skip_nl()
        while not self.match(END) and not self.match(EOF):
            s = self.stat()
            if s is not None:
                body.append(s)
            self.skip_nl()

        self.consume(END)
        if self.match(NEWLINE):
            self.consume()

        return FuncDefContext(name, params, body)

    # Parsea return expr.
    def return_stat(self):
        self.consume(RETURN)
        e = self.expr()
        if self.match(NEWLINE):
            self.consume()
        return ReturnContext(e)

    # Parsea if con bloque else opcional.
    def if_stat(self):
        self.consume(IF)
        cond = self.condition()
        if self.match(NEWLINE):
            self.consume()

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
            if self.match(NEWLINE):
                self.consume()
            self.skip_nl()
            while not self.match(END) and not self.match(EOF):
                s = self.stat()
                if s is not None:
                    else_stats.append(s)
                self.skip_nl()

        self.consume(END)
        if self.match(NEWLINE):
            self.consume()

        return IfContext(cond, then_stats, else_stats)

    # Parsea while ... end.
    def while_stat(self):
        self.consume(WHILE)
        cond = self.condition()
        if self.match(NEWLINE):
            self.consume()

        body = []
        self.skip_nl()
        while not self.match(END) and not self.match(EOF):
            s = self.stat()
            if s is not None:
                body.append(s)
            self.skip_nl()

        self.consume(END)
        if self.match(NEWLINE):
            self.consume()

        return WhileContext(cond, body)

    # Parsea for var = inicio to fin ... end.
    def for_stat(self):
        self.consume(FOR)
        var = self.consume(ID).text
        self.consume()          # consume '='
        start = self.expr()
        self.consume(TO)
        end_ = self.expr()
        if self.match(NEWLINE):
            self.consume()

        body = []
        self.skip_nl()
        while not self.match(END) and not self.match(EOF):
            s = self.stat()
            if s is not None:
                body.append(s)
            self.skip_nl()

        self.consume(END)
        if self.match(NEWLINE):
            self.consume()

        return ForContext(var, start, end_, body)

    # Parsea una condición con operador relacional.
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

    # Punto de entrada para parsear expresiones.
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
            op = self.consume()
            right = self._pow()
            return PowContext(left, op, right)
        return left

    def _primary(self):
        tok = self.current()

        if tok.type == SUB:
            self.consume()
            return UnaryMinusContext(self._primary())

        if tok.type == LBRACKET:
            self.consume()
            elements = []
            if not self.match(RBRACKET):
                elements.append(self.expr())
                while self.match(COMMA):
                    self.consume()
                    elements.append(self.expr())
            self.consume(RBRACKET)
            return ArrayLiteralContext(elements)

        if tok.type == INT:
            self.consume()
            return IntContext(int(tok.text))

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
