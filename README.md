# DeepLang-proyecto

DeepLang es un lenguaje de programación de dominio específico (DSL) diseñado para realizar operaciones aritméticas. Está implementado en Python siguiendo el patrón de diseño Visitor.

## Archivos del proyecto

| Archivo | Descripción |
|---|---|
| `DeepLang.g4` | La gramática formal — define la sintaxis del lenguaje |
| `lexer.py` | Lee el código fuente y lo convierte en tokens |
| `parser.py` | Toma los tokens y construye el árbol de sintaxis |
| `EvalVisitor.py` | Recorre el árbol y ejecuta las operaciones |
| `DeepLang.py` | Punto de entrada — conecta todos los componentes |
| `t.dl` | Archivo de prueba con ejemplos |

## Cómo ejecutar

### Con archivo
```bash
python3 DeepLang.py t.dl
```

### Modo interactivo
```bash
python3 DeepLang.py
```

## Qué puede hacer

```
5 + 3        → suma
10 - 4       → resta
6 * 7        → multiplicación
20 / 4       → división
(1+2)*3      → paréntesis
a = 5        → asignar variable
a + 10       → usar variable
```

## Ejemplo

Un ejemplo que se puede hacer en deeplang por el momento.
```
193
a = 5
b = 6
a+b*2
(1+2)*3
```

Salida:
```
193
17
9
```

## Flujo del programa

```
Código fuente (.dl)
       ↓
   lexer.py       →  convierte texto en tokens
       ↓
   parser.py      →  construye el árbol de sintaxis
       ↓
   EvalVisitor.py →  ejecuta las operaciones
       ↓
   Resultado en pantalla
```
