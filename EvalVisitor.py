# EvalVisitor.py
from lexer import MUL, DIV, ADD, SUB, POW, EQ, NEQ, LT, GT, LTE, GTE
from parser import (
    ProgContext, AssignContext, PrintExprContext, PrintContext,
    MulDivContext, AddSubContext, PowContext, UnaryMinusContext,
    IntContext, FloatContext, StringContext, IdContext, ParensContext,
    IfContext, ConditionContext, WhileContext, ForContext,
    ArrayLiteralContext, ArrayAccessContext, ArrayAssignContext,
    FuncDefContext, FuncCallContext, ReturnContext,
)
from deeplang_filelib     import DeepLangFileLib
from deeplang_graficaslib  import DeepLangGraficasLib
from deeplang_matriceslib  import DeepLangMatricesLib


class ReturnSignal(Exception):
    """Excepción interna para cortar la ejecución al hacer return."""
    def __init__(self, value):
        self.value = value


class EvalVisitor:
    PI     = 3.141592653589793
    TWO_PI = 2 * PI

    def __init__(self):
        self.memory    = {}
        self.functions = {}
        filelib    = DeepLangFileLib()
        graficaslib  = DeepLangGraficasLib()
        matriceslib  = DeepLangMatricesLib()
        self.builtins = {
            # ── Trigonométricas ───────────────────────────────
            'sen':        (self._sin,  1),
            'sin':        (self._sin,  1),
            'cos':        (self._cos,  1),
            'tan':        (self._tan,  1),
            'cosecante':  (self._csc,  1),
            'csc':        (self._csc,  1),
            'secante':    (self._sec,  1),
            'sec':        (self._sec,  1),
            'cotangente': (self._cot,  1),
            'cot':        (self._cot,  1),
            'ctg':        (self._cot,  1),
            # ── Matemáticas generales ─────────────────────────
            'modulo':     (self._mod,   2),
            'mod':        (self._mod,   2),
            'raiz':       (self._sqrt,  1),
            'abs':        (self._abs,   1),
            'redondear':  (self._round, 2),
            'piso':       (self._floor, 1),
            'techo':      (self._ceil,  1),
            'entero':     (self._int,   1),
            'flotante':   (self._float, 1),
            'log':        (self._log,   1),
            'log10':      (self._log10, 1),
            'exp':        (self._exp,   1),
            # ── Archivos de texto ─────────────────────────────
            'leerarchivo':     (filelib.leerarchivo,     1),
            'leerlineas':      (filelib.leerlineas,      1),
            'escribirarchivo': (filelib.escribirarchivo, 2),
            'agregararchivo':  (filelib.agregararchivo,  2),
            # ── CSV ───────────────────────────────────────────
            'leercsv':         (filelib.leercsv,         1),
            'leercsv_datos':   (filelib.leercsv_datos,   1),
            'leercsv_columna': (filelib.leercsv_columna, 2),
            'escribircsv':     (filelib.escribircsv,     2),
            # ── Gráficas ASCII ────────────────────────────────
            'grafica_barras':     (graficaslib.grafica_barras,     3),
            'grafica_barras_v':   (graficaslib.grafica_barras_v,   3),
            'grafica_linea':      (graficaslib.grafica_linea,      4),
            'grafica_dispersion': (graficaslib.grafica_dispersion, 5),
            'histograma':         (graficaslib.histograma,         4),
            'grafica_pastel':     (graficaslib.grafica_pastel,     2),
            'grafica_funcion':    (graficaslib.grafica_funcion,    6),
            # ── Matrices ──────────────────────────────────────
            'mat_get':         (matriceslib.mat_get,         3),
            'mat_idx':         (matriceslib.mat_idx,         3),
            'mat_ceros':       (matriceslib.mat_ceros,       2),
            'mat_identidad':   (matriceslib.mat_identidad,   1),
            'mat_suma':        (matriceslib.mat_suma,        4),
            'mat_resta':       (matriceslib.mat_resta,       4),
            'mat_escalar':     (matriceslib.mat_escalar,     4),
            'mat_mul':         (matriceslib.mat_mul,         5),
            'mat_transpuesta': (matriceslib.mat_transpuesta, 3),
            'mat_traza':       (matriceslib.mat_traza,       2),
            'mat_inversa':     (matriceslib.mat_inversa,     2),
            'mat_det':         (matriceslib.mat_det,         2),
            'mat_imprimir':    (matriceslib.mat_imprimir,    3),
            'vec_dot':         (matriceslib.vec_dot,         3),
            'vec_norma':       (matriceslib.vec_norma,       2),
            # ── Arreglos ──────────────────────────────────────

            'longitud':   (self._len, 1),
            'len':        (self._len, 1),
        }

    # ─── Normalización de ángulos ────────────────────────────
    def _normalize_angle(self, x):
        x = x % self.TWO_PI
        if x > self.PI:
            x -= self.TWO_PI
        return x

    # ─── Trigonométricas (series de Taylor) ──────────────────
    def _sin(self, x):
        x      = self._normalize_angle(float(x))
        term   = x
        result = x
        for n in range(1, 12):
            term *= -1 * x * x / ((2 * n) * (2 * n + 1))
            result += term
        return result

    def _cos(self, x):
        x      = self._normalize_angle(float(x))
        term   = 1.0
        result = 1.0
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

    # ─── Matemáticas generales ───────────────────────────────
    def _mod(self, a, b):
        if b == 0:
            raise ZeroDivisionError("modulo(a, b) indefinida cuando b = 0")
        return a % b

    def _sqrt(self, x):
        if x < 0:
            raise RuntimeError("raiz(x) no está definida para x < 0")
        if x == 0:
            return 0.0
        guess = float(x) if x >= 1 else 1.0
        for _ in range(40):
            guess = 0.5 * (guess + x / guess)
        return guess

    def _abs(self, x):
        return x if x >= 0 else -x

    def _round(self, x, decimals):
        factor = 1.0
        d = int(decimals)
        for _ in range(d):
            factor *= 10
        if x >= 0:
            return int(x * factor + 0.5) / factor
        return -int(-x * factor + 0.5) / factor

    def _floor(self, x):
        if isinstance(x, int):
            return x
        n = int(x)
        return n if x >= 0 or x == n else n - 1

    def _ceil(self, x):
        if isinstance(x, int):
            return x
        n = int(x)
        return n if x <= 0 or x == n else n + 1

    def _int(self, x):
        return int(x)

    def _float(self, x):
        return float(x)

    def _log(self, x):
        if x <= 0:
            raise RuntimeError("log(x) no definido para x <= 0")
        k   = 0
        m   = float(x)
        ln2 = 0.6931471805599453
        while m >= 2.0:
            m /= 2.0
            k += 1
        while m < 1.0:
            m *= 2.0
            k -= 1
        y      = m - 1.0
        term   = y
        result = y
        for n in range(2, 60):
            term *= -y * (n - 1) / n
            result += term
            if abs(term) < 1e-15:
                break
        return result + k * ln2

    def _log10(self, x):
        return self._log(x) / self._log(10.0)

    def _exp(self, x):
        x      = float(x)
        term   = 1.0
        result = 1.0
        for n in range(1, 80):
            term *= x / n
            result += term
            if abs(term) < 1e-15:
                break
        return result

    def _len(self, arr):
        if not isinstance(arr, list):
            raise RuntimeError("longitud() espera un arreglo")
        return len(arr)

    # ─── Dispatcher principal ────────────────────────────────
    def visit(self, ctx):
        if isinstance(ctx, ProgContext):         return self.visitProg(ctx)
        if isinstance(ctx, AssignContext):       return self.visitAssign(ctx)
        if isinstance(ctx, PrintExprContext):    return self.visitPrintExpr(ctx)
        if isinstance(ctx, PrintContext):        return self.visitPrint(ctx)
        if isinstance(ctx, MulDivContext):       return self.visitMulDiv(ctx)
        if isinstance(ctx, PowContext):          return self.visitPow(ctx)
        if isinstance(ctx, AddSubContext):       return self.visitAddSub(ctx)
        if isinstance(ctx, UnaryMinusContext):   return self.visitUnaryMinus(ctx)
        if isinstance(ctx, IntContext):          return self.visitInt(ctx)
        if isinstance(ctx, FloatContext):        return self.visitFloat(ctx)
        if isinstance(ctx, StringContext):       return self.visitString(ctx)
        if isinstance(ctx, IdContext):           return self.visitId(ctx)
        if isinstance(ctx, ParensContext):       return self.visitParens(ctx)
        if isinstance(ctx, IfContext):           return self.visitIf(ctx)
        if isinstance(ctx, ConditionContext):    return self.visitCondition(ctx)
        if isinstance(ctx, WhileContext):        return self.visitWhile(ctx)
        if isinstance(ctx, ForContext):          return self.visitFor(ctx)
        if isinstance(ctx, ArrayLiteralContext): return self.visitArrayLiteral(ctx)
        if isinstance(ctx, ArrayAccessContext):  return self.visitArrayAccess(ctx)
        if isinstance(ctx, ArrayAssignContext):  return self.visitArrayAssign(ctx)
        if isinstance(ctx, FuncDefContext):      return self.visitFuncDef(ctx)
        if isinstance(ctx, FuncCallContext):     return self.visitFuncCall(ctx)
        if isinstance(ctx, ReturnContext):       return self.visitReturn(ctx)
        raise RuntimeError(f"Nodo desconocido: {type(ctx)}")

    # ─── Visitores ───────────────────────────────────────────
    def visitProg(self, ctx):
        for stat in ctx.stats:
            self.visit(stat)

    def visitAssign(self, ctx):
        value = self.visit(ctx.expr)
        self.memory[ctx.name] = value
        return value

    def visitPrintExpr(self, ctx):
        return self.visit(ctx.expr)

    def visitPrint(self, ctx):
        value = self.visit(ctx.expr)
        if isinstance(value, float) and value == int(value):
            print(int(value))
        elif isinstance(value, float):
            s = f"{value:.10f}".rstrip('0').rstrip('.')
            print(s)
        else:
            print(value)
        return value

    def visitMulDiv(self, ctx):
        left  = self.visit(ctx.left)
        right = self.visit(ctx.right)
        if ctx.op.type == MUL:
            return left * right
        if right == 0:
            raise ZeroDivisionError("No se puede dividir entre cero")
        if isinstance(left, float) or isinstance(right, float):
            return left / right
        return left // right

    def visitPow(self, ctx):
        left  = self.visit(ctx.left)
        right = self.visit(ctx.right)
        return left ** right

    def visitAddSub(self, ctx):
        left  = self.visit(ctx.left)
        right = self.visit(ctx.right)
        if ctx.op.type == ADD:
            if isinstance(left, str) or isinstance(right, str):
                def _fmt(v):
                    if isinstance(v, float) and v == int(v):
                        return str(int(v))
                    if isinstance(v, float):
                        return f"{v:.10f}".rstrip("0").rstrip(".")
                    return str(v)
                return _fmt(left) + _fmt(right)
            return left + right
        return left - right

    def visitUnaryMinus(self, ctx):
        return -self.visit(ctx.expr)

    def visitInt(self, ctx):
        return ctx.value

    def visitFloat(self, ctx):
        return ctx.value

    def visitString(self, ctx):
        return ctx.value

    def visitId(self, ctx):
        if ctx.name in self.memory:
            return self.memory[ctx.name]
        return 0

    def visitParens(self, ctx):
        return self.visit(ctx.expr)

    def visitIf(self, ctx):
        if self.visit(ctx.cond):
            for stat in ctx.then_stats:
                self.visit(stat)
        else:
            for stat in ctx.else_stats:
                self.visit(stat)

    def visitWhile(self, ctx):
        while self.visit(ctx.cond):
            for stat in ctx.stats:
                self.visit(stat)

    def visitFor(self, ctx):
        start = self.visit(ctx.start)
        end_  = self.visit(ctx.end_)
        self.memory[ctx.var] = start
        while self.memory[ctx.var] <= end_:
            for stat in ctx.body:
                self.visit(stat)
            self.memory[ctx.var] += 1

    def visitArrayLiteral(self, ctx):
        return [self.visit(e) for e in ctx.elements]

    def visitArrayAccess(self, ctx):
        arr = self.memory.get(ctx.name)
        if not isinstance(arr, list):
            raise RuntimeError(f"'{ctx.name}' no es un arreglo")
        index = self.visit(ctx.index)
        if index < 0 or index >= len(arr):
            raise RuntimeError(
                f"Índice {index} fuera de rango para '{ctx.name}' (tamaño {len(arr)})"
            )
        return arr[index]

    def visitArrayAssign(self, ctx):
        arr = self.memory.get(ctx.name)
        if not isinstance(arr, list):
            raise RuntimeError(f"'{ctx.name}' no es un arreglo")
        index = self.visit(ctx.index)
        if index < 0:
            raise RuntimeError(f"Índice negativo {index} para '{ctx.name}'")
        value = self.visit(ctx.expr)
        if index == len(arr):
            arr.append(value)
        elif index < len(arr):
            arr[index] = value
        else:
            raise RuntimeError(
                f"Índice {index} fuera de rango para '{ctx.name}' "
                f"(tamaño {len(arr)}). Siguiente índice válido: {len(arr)}"
            )
        return value

    def visitFuncDef(self, ctx):
        self.functions[ctx.name] = ctx

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

        valores          = [self.visit(a) for a in ctx.args]
        memoria_anterior = self.memory.copy()
        self.memory      = memoria_anterior.copy()
        for param, val in zip(func.params, valores):
            self.memory[param] = val

        resultado = 0
        try:
            for stat in func.body:
                self.visit(stat)
        except ReturnSignal as r:
            resultado = r.value

        self.memory = memoria_anterior
        return resultado

    def visitReturn(self, ctx):
        raise ReturnSignal(self.visit(ctx.expr))

    def visitCondition(self, ctx):
        left  = self.visit(ctx.left)
        right = self.visit(ctx.right)
        op    = ctx.op.type
        if op == EQ:  return left == right
        if op == NEQ: return left != right
        if op == LT:  return left < right
        if op == GT:  return left > right
        if op == LTE: return left <= right
        if op == GTE: return left >= right
        raise RuntimeError(f"Operador de comparación desconocido: {op}")
