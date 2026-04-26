grammar DeepLang;

prog:   stat+ ;

stat:   expr NEWLINE                                        # printExpr
    |   ID '=' expr NEWLINE                                 # assign
    |   ID '[' expr ']' '=' expr NEWLINE                    # arrayAssign
    |   'if' condition NEWLINE stat* ('else' NEWLINE stat*)? 'end' NEWLINE   # ifStat
    |   'while' condition NEWLINE stat* 'end' NEWLINE       # whileStat
    |   'fun' ID '(' params? ')' NEWLINE stat* 'end' NEWLINE # funDef
    |   'return' expr NEWLINE                               # returnStat
    |   'print' '(' expr ')' NEWLINE                        # printStat
    |   NEWLINE                                             # blank
    ;

params: ID (',' ID)* ;

condition: expr compOp expr ;

compOp: '==' | '!=' | '<' | '>' | '<=' | '>=' ;

expr:   expr op=('*'|'/') expr      # MulDiv
    |   expr op=('+'|'-') expr      # AddSub
    |   ID '(' args? ')'            # funcCall
    |   ID '[' expr ']'             # arrayAccess
    |   '[' (expr (',' expr)*)? ']' # arrayLiteral
    |   INT                         # int
    |   ID                          # id
    |   '(' expr ')'                # parens
    ;

args: expr (',' expr)* ;

// ── Tokens ─────────────────────────────────────────────────
MUL     : '*' ;
DIV     : '/' ;
ADD     : '+' ;
SUB     : '-' ;
EQ      : '==' ;
NEQ     : '!=' ;
LT      : '<' ;
GT      : '>' ;
LTE     : '<=' ;
GTE     : '>=' ;
LBRACK  : '[' ;
RBRACK  : ']' ;
COMMA   : ',' ;
ID      : [a-zA-Z]+ ;
INT     : [0-9]+ ;
NEWLINE : '\r'? '\n' ;
WS      : [ \t]+ -> skip ;
