# EvalVisitor.py

from lexer  import MUL, DIV, ADD, SUB
from parser import (
    ProgContext, AssignContext, PrintExprContext,
    MulDivContext, AddSubContext,
    IntContext, IdContext, ParensContext,
)


class EvalVisitor:

    def __init__(self):
        # "memory" for our calculator; variable/value pairs go here
        self.memory = {}

    def visit(self, ctx):
        if isinstance(ctx, ProgContext):       return self.visitProg(ctx)
        if isinstance(ctx, AssignContext):     return self.visitAssign(ctx)
        if isinstance(ctx, PrintExprContext):  return self.visitPrintExpr(ctx)
        if isinstance(ctx, MulDivContext):     return self.visitMulDiv(ctx)
        if isinstance(ctx, AddSubContext):     return self.visitAddSub(ctx)
        if isinstance(ctx, IntContext):        return self.visitInt(ctx)
        if isinstance(ctx, IdContext):         return self.visitId(ctx)
        if isinstance(ctx, ParensContext):     return self.visitParens(ctx)
        raise RuntimeError(f"Nodo desconocido: {type(ctx)}")

    # ── visitProg ──────────────────────────────────────────

    def visitProg(self, ctx):
        for stat in ctx.stats:
            self.visit(stat)

    # ── ID '=' expr NEWLINE ────────────────────────────────

    def visitAssign(self, ctx):
        # String id
        id_ = ctx.name

        # int value 
        value = self.visit(ctx.expr)

        # memory.put(id, value);
        self.memory[id_] = value

        return value

    # ── expr NEWLINE ───────────────────────────────────────

    def visitPrintExpr(self, ctx):
        # Integer value 
        value = self.visit(ctx.expr)

        # System.out.println(value);
        print(value)

        return 0   

    # ── expr op=('*'|'/') expr ────────────────────────────

    def visitMulDiv(self, ctx):
        # int left  
        left  = self.visit(ctx.left)

        # int right
        right = self.visit(ctx.right)

    
        if ctx.op.type == MUL:
            return left * right

        # return left / right;  
        if right == 0:
            raise ZeroDivisionError("No se puede dividir entre cero")
        return left // right

    # ── expr op=('+'|'-') expr ────────────────────────────

    def visitAddSub(self, ctx):
        # int left  = visit(ctx.expr(0));
        left  = self.visit(ctx.left)

        # int right = visit(ctx.expr(1));
        right = self.visit(ctx.right)

        # if (ctx.op.getType() == LabeledExprParser.ADD) return left + right;
        if ctx.op.type == ADD:
            return left + right

        # return left - right;
        return left - right

    # ── INT ────────────────────────────────────────────────

    def visitInt(self, ctx):
        # return Integer.valueOf(ctx.INT().getText());
        return ctx.value

    # ── ID ─────────────────────────────────────────────────

    def visitId(self, ctx):
        # String id = ctx.ID().getText();
        id_ = ctx.name

        # if (memory.containsKey(id)) return memory.get(id);
        if id_ in self.memory:
            return self.memory[id_]

        return 0    # return 0 si no existe

    # ── '(' expr ')' ───────────────────────────────────────

    def visitParens(self, ctx):
        # return visit(ctx.expr());
        return self.visit(ctx.expr)
