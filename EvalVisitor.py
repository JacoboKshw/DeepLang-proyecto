# EvalVisitor.py

from lexer  import MUL, DIV, ADD, SUB, POW, EQ, NEQ, LT, GT, LTE, GTE
from parser import (
    ProgContext, AssignContext, PrintExprContext, PrintContext,
    MulDivContext, AddSubContext, PowContext,
    IntContext, IdContext, ParensContext,
    IfContext, ConditionContext, WhileContext,
    ArrayLiteralContext, ArrayAccessContext, ArrayAssignContext,
    FuncDefContext, FuncCallContext, ReturnContext,
)


class ReturnSignal(Exception):
    """Se lanza con return para salir del cuerpo de una función."""
    def __init__(self, value):
        self.value = value


class EvalVisitor:

    PI = 3.141592653589793
    TWO_PI = 2 * PI

    def __init__(self):
        self.memory    = {}   # variables globales
        self.functions = {}   # nombre → FuncDefContext
        self.builtins  = {
            'sen': (self._sin, 1),
            'sin': (self._sin, 1),
            'cos': (self._cos, 1),
            'tan': (self._tan, 1),
            'cosecante': (self._csc, 1),
            'csc': (self._csc, 1),
            'secante': (self._sec, 1),
            'sec': (self._sec, 1),
            'cotangente': (self._cot, 1),
            'cot': (self._cot, 1),
            'ctg': (self._cot, 1),
            'modulo': (self._mod, 2),
            'mod': (self._mod, 2),
        }

    def _normalize_angle(self, x):
        """Reduce x al rango [-pi, pi] para mejorar convergencia."""
        x = x % self.TWO_PI
        if x > self.PI:
            x -= self.TWO_PI
        return x

    def _sin(self, x):
        x = self._normalize_angle(x)
        term = x
        result = x
        # Serie de Taylor de sin(x)
        for n in range(1, 12):
            term *= -1 * x * x / ((2 * n) * (2 * n + 1))
            result += term
        return result

    def _cos(self, x):
        x = self._normalize_angle(x)
        term = 1.0
        result = 1.0
        # Serie de Taylor de cos(x)
        for n in range(1, 12):
            term *= -1 * x * x / ((2 * n - 1) * (2 * n))
            result += term
        return result

    def _tan(self, x):
        c = self._cos(x)
        if -1e-12 < c < 1e-12:
            raise ZeroDivisionError("tan(x) indefinida cuando cos(x) = 0")
        return self._sin(x) / c

    def _sec(self, x):
        c = self._cos(x)
        if -1e-12 < c < 1e-12:
            raise ZeroDivisionError("sec(x) indefinida cuando cos(x) = 0")
        return 1.0 / c

    def _csc(self, x):
        s = self._sin(x)
        if -1e-12 < s < 1e-12:
            raise ZeroDivisionError("cosecante(x) indefinida cuando sen(x) = 0")
        return 1.0 / s

    def _cot(self, x):
        s = self._sin(x)
        if -1e-12 < s < 1e-12:
            raise ZeroDivisionError("cotangente(x) indefinida cuando sen(x) = 0")
        return self._cos(x) / s

    def _mod(self, a, b):
        if b == 0:
            raise ZeroDivisionError("modulo(a, b) indefinida cuando b = 0")
        return a % b

    def visit(self, ctx):
        if isinstance(ctx, ProgContext):       return self.visitProg(ctx)
        if isinstance(ctx, AssignContext):     return self.visitAssign(ctx)
        if isinstance(ctx, PrintExprContext):  return self.visitPrintExpr(ctx)
        if isinstance(ctx, PrintContext):      return self.visitPrint(ctx)
        if isinstance(ctx, MulDivContext):     return self.visitMulDiv(ctx)
        if isinstance(ctx, PowContext):        return self.visitPow(ctx)
        if isinstance(ctx, AddSubContext):     return self.visitAddSub(ctx)
        if isinstance(ctx, IntContext):        return self.visitInt(ctx)
        if isinstance(ctx, IdContext):         return self.visitId(ctx)
        if isinstance(ctx, ParensContext):     return self.visitParens(ctx)
        if isinstance(ctx, IfContext):            return self.visitIf(ctx)
        if isinstance(ctx, ConditionContext):     return self.visitCondition(ctx)
        if isinstance(ctx, WhileContext):         return self.visitWhile(ctx)
        if isinstance(ctx, ArrayLiteralContext):  return self.visitArrayLiteral(ctx)
        if isinstance(ctx, ArrayAccessContext):   return self.visitArrayAccess(ctx)
        if isinstance(ctx, ArrayAssignContext):   return self.visitArrayAssign(ctx)
        if isinstance(ctx, FuncDefContext):        return self.visitFuncDef(ctx)
        if isinstance(ctx, FuncCallContext):       return self.visitFuncCall(ctx)
        if isinstance(ctx, ReturnContext):         return self.visitReturn(ctx)
        raise RuntimeError(f"Nodo desconocido: {type(ctx)}")

    # ── visitProg ──────────────────────────────────────────

    def visitProg(self, ctx):
        for stat in ctx.stats:
            self.visit(stat)

    # ── ID '=' expr NEWLINE ────────────────────────────────

    def visitAssign(self, ctx):
        id_   = ctx.name
        value = self.visit(ctx.expr)
        self.memory[id_] = value
        return value

    # ── expr sola — evalúa pero NO imprime ────────────────

    def visitPrintExpr(self, ctx):
        return self.visit(ctx.expr)

    # ── print(expr) — sí imprime ───────────────────────────

    def visitPrint(self, ctx):
        value = self.visit(ctx.expr)
        print(value)
        return value

    # ── expr op=('*'|'/') expr ────────────────────────────

    def visitMulDiv(self, ctx):
        left  = self.visit(ctx.left)
        right = self.visit(ctx.right)
        if ctx.op.type == MUL:
            return left * right
        if right == 0:
            raise ZeroDivisionError("No se puede dividir entre cero")
        return left // right

    # ── expr '^' expr ──────────────────────────────────────

    def visitPow(self, ctx):
        left  = self.visit(ctx.left)
        right = self.visit(ctx.right)
        if isinstance(right, float) and not right.is_integer():
            raise RuntimeError("El exponente en x^y debe ser entero")
        return left ** int(right)

    # ── expr op=('+'|'-') expr ────────────────────────────

    def visitAddSub(self, ctx):
        left  = self.visit(ctx.left)
        right = self.visit(ctx.right)
        if ctx.op.type == ADD:
            return left + right
        return left - right

    # ── INT ────────────────────────────────────────────────

    def visitInt(self, ctx):
        return ctx.value

    # ── ID ─────────────────────────────────────────────────

    def visitId(self, ctx):
        id_ = ctx.name
        if id_ in self.memory:
            return self.memory[id_]
        return 0

    # ── '(' expr ')' ───────────────────────────────────────

    def visitParens(self, ctx):
        return self.visit(ctx.expr)

    # ── if cond then ... [else ...] end ────────────────────

    def visitIf(self, ctx):
        # Evalúa la condición: True o False
        cond_result = self.visit(ctx.cond)

        if cond_result:
            for stat in ctx.then_stats:
                self.visit(stat)
        else:
            for stat in ctx.else_stats:
                self.visit(stat)

    # ── while cond ... end ─────────────────────────────────

    def visitWhile(self, ctx):
        while self.visit(ctx.cond):
            for stat in ctx.stats:
                self.visit(stat)

    # ── [e1, e2, e3] ───────────────────────────────────────

    def visitArrayLiteral(self, ctx):
        return [self.visit(e) for e in ctx.elements]

    # ── lista[i] ───────────────────────────────────────────

    def visitArrayAccess(self, ctx):
        arr   = self.memory.get(ctx.name)
        if not isinstance(arr, list):
            raise RuntimeError(f"'{ctx.name}' no es un arreglo")
        index = self.visit(ctx.index)
        if index < 0 or index >= len(arr):
            raise RuntimeError(
                f"Índice {index} fuera de rango para '{ctx.name}' (tamaño {len(arr)})"
            )
        return arr[index]

    # ── lista[i] = expr ────────────────────────────────────

    def visitArrayAssign(self, ctx):
        arr = self.memory.get(ctx.name)
        if not isinstance(arr, list):
            raise RuntimeError(f"'{ctx.name}' no es un arreglo")
        index = self.visit(ctx.index)
        if index < 0 or index >= len(arr):
            raise RuntimeError(
                f"Índice {index} fuera de rango para '{ctx.name}' (tamaño {len(arr)})"
            )
        value = self.visit(ctx.expr)
        arr[index] = value
        return value

    # ── fun nombre(params) body end ────────────────────────

    def visitFuncDef(self, ctx):
        # solo guarda la definición, no ejecuta nada
        self.functions[ctx.name] = ctx

    # ── nombre(args) ───────────────────────────────────────

    def visitFuncCall(self, ctx):
        if ctx.name in self.builtins:
            builtin_fn, arity = self.builtins[ctx.name]
            if len(ctx.args) != arity:
                raise RuntimeError(
                    f"'{ctx.name}' espera {arity} argumento(s), recibió {len(ctx.args)}"
                )
            values = [self.visit(arg) for arg in ctx.args]
            return builtin_fn(*values)

        if ctx.name not in self.functions:
            raise RuntimeError(f"Función '{ctx.name}' no definida")

        func = self.functions[ctx.name]

        if len(ctx.args) != len(func.params):
            raise RuntimeError(
                f"'{ctx.name}' espera {len(func.params)} argumento(s), "
                f"recibió {len(ctx.args)}"
            )

        # evaluar argumentos en el ámbito actual
        valores = [self.visit(a) for a in ctx.args]

        # guardar memoria actual y crear ámbito local
        memoria_anterior = self.memory.copy()
        self.memory = memoria_anterior.copy()   # hereda variables globales
        for param, val in zip(func.params, valores):
            self.memory[param] = val

        # ejecutar cuerpo — capturar return
        resultado = 0
        try:
            for stat in func.body:
                self.visit(stat)
        except ReturnSignal as r:
            resultado = r.value

        # restaurar memoria anterior
        self.memory = memoria_anterior
        return resultado

    # ── return expr ────────────────────────────────────────

    def visitReturn(self, ctx):
        value = self.visit(ctx.expr)
        raise ReturnSignal(value)

    # ── Condición: expr op expr ────────────────────────────

    def visitCondition(self, ctx):
        left  = self.visit(ctx.left)
        right = self.visit(ctx.right)
        op    = ctx.op.type

        if op == EQ:  return left == right
        if op == NEQ: return left != right
        if op == LT:  return left <  right
        if op == GT:  return left >  right
        if op == LTE: return left <= right
        if op == GTE: return left >= right

        raise RuntimeError(f"Operador de comparación desconocido: {op}")
