# parser.py

from lexer import (
    MUL, DIV, ADD, SUB, ID, INT, NEWLINE, LPAREN, RPAREN, EOF,
    IF, ELSE, END, WHILE,
    EQ, NEQ, LT, GT, LTE, GTE,
    LBRACKET, RBRACKET, COMMA,
    PRINT, FUN, RETURN,
)


class ParseError(Exception):
    pass


# ── Nodos del árbol ────────────────────────────────────────

class ProgContext:
    """prog: stat+"""
    def __init__(self, stats):
        self.stats = stats


class AssignContext:
    """ # assign"""
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
    """# MulDiv"""
    def __init__(self, left, op, right):
        self.left  = left
        self.op    = op    
        self.right = right


class AddSubContext:
    """# AddSub"""
    def __init__(self, left, op, right):
        self.left  = left
        self.op    = op    
        self.right = right


class IntContext:
    """# int"""
    def __init__(self, value):
        self.value = value


class IdContext:
    """# id"""
    def __init__(self, name):
        self.name = name    


class ParensContext:
    """# parens"""
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


class ArrayLiteralContext:
    """lista = [1, 2, 3]"""
    def __init__(self, elements):
        self.elements = elements   # lista de expr


class ArrayAccessContext:
    """lista[0]  — como expresión"""
    def __init__(self, name, index):
        self.name  = name          # str
        self.index = index         # expr


class ArrayAssignContext:
    """lista[0] = 5"""
    def __init__(self, name, index, expr):
        self.name  = name          # str
        self.index = index         # expr
        self.expr  = expr          # expr


class FuncDefContext:
    """
    fun nombre(a, b)
        stats
    end
    """
    def __init__(self, name, params, body):
        self.name   = name    # str
        self.params = params  # lista de str
        self.body   = body    # lista de stat


class FuncCallContext:
    """nombre(expr, expr)  — como expresión"""
    def __init__(self, name, args):
        self.name = name   # str
        self.args = args   # lista de expr


class ReturnContext:
    """return expr"""
    def __init__(self, expr):
        self.expr = expr


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

        # if cond ... [else ...] end
        if self.match(IF):
            return self.if_stat()

        # while cond ... end
        if self.match(WHILE):
            return self.while_stat()

        # print(expr)
        if self.match(PRINT):
            return self.print_stat()

        # fun nombre(params) ... end
        if self.match(FUN):
            return self.fun_def()

        # return expr
        if self.match(RETURN):
            return self.return_stat()

        # assign:  ID '=' expr NEWLINE
        #          ID '[' expr ']' '=' expr NEWLINE
        if self.match(ID) and self.pos + 1 < len(self.tokens):
            next_tok = self.tokens[self.pos + 1]

            # lista[i] = expr  — busca hacia adelante el ']' y verifica que sigue '='
            if next_tok.type == LBRACKET:
                # buscar el ] correspondiente para ver si sigue =
                depth = 1
                look  = self.pos + 2
                while look < len(self.tokens) and depth > 0:
                    t = self.tokens[look]
                    if t.type == LBRACKET: depth += 1
                    if t.type == RBRACKET: depth -= 1
                    look += 1
                # tokens[look-1] es el ']'; tokens[look] debe ser '='
                is_assign = (look < len(self.tokens) and
                             self.tokens[look].text == '=')
                if is_assign:
                    name = self.consume(ID).text
                    self.consume(LBRACKET)
                    index = self.expr()
                    self.consume(RBRACKET)
                    self.consume()           # consume '='
                    expr = self.expr()
                    if self.match(NEWLINE):
                        self.consume()
                    return ArrayAssignContext(name, index, expr)
                # si no hay '=', cae al caso expr NEWLINE de abajo

            # var = expr
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

    # ── print(expr) ────────────────────────────────────────
    def print_stat(self):
        self.consume(PRINT)
        self.consume(LPAREN)
        e = self.expr()
        self.consume(RPAREN)
        if self.match(NEWLINE):
            self.consume()
        return PrintContext(e)

    # ── fun nombre(a, b) body end ──────────────────────────
    def fun_def(self):
        self.consume(FUN)
        name = self.consume(ID).text
        self.consume(LPAREN)

        # parámetros
        params = []
        if not self.match(RPAREN):
            params.append(self.consume(ID).text)
            while self.match(COMMA):
                self.consume()
                params.append(self.consume(ID).text)
        self.consume(RPAREN)
        if self.match(NEWLINE):
            self.consume()

        # cuerpo
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

    # ── return expr ────────────────────────────────────────
    def return_stat(self):
        self.consume(RETURN)
        e = self.expr()
        if self.match(NEWLINE):
            self.consume()
        return ReturnContext(e)

    # ── if cond body [else body] end ──────────────────────
    def if_stat(self):
        self.consume(IF)
        cond = self.condition()
        if self.match(NEWLINE):
            self.consume()

        # bloque then
        then_stats = []
        self.skip_nl()
        while not self.match(ELSE) and not self.match(END) and not self.match(EOF):
            s = self.stat()
            if s is not None:
                then_stats.append(s)
            self.skip_nl()

        # bloque else (opcional)
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

    # ── while cond body end ────────────────────────────────
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

    # ── condition: expr compOp expr ────────────────────────
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

        # '[' expr (',' expr)* ']'   →  literal de arreglo
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

        # ID  o  ID '[' expr ']'  o  ID '(' args ')'
        if tok.type == ID:
            name = self.consume().text
            # llamada a función: nombre(args)
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
            # acceso a arreglo: nombre[index]
            if self.match(LBRACKET):
                self.consume()
                index = self.expr()
                self.consume(RBRACKET)
                return ArrayAccessContext(name, index)
            return IdContext(name)

        raise ParseError(
            f"Línea {tok.line}: token inesperado '{tok.type}' ({tok.text!r})"
        )
