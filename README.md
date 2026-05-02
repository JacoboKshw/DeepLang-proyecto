Jacobo Mondragon

Julian Gomez 

# DeepLang

DeepLang es un lenguaje de programación de dominio específico (DSL) diseñado para realizar operaciones aritméticas y lógicas. Está implementado en Python siguiendo el patrón de diseño Visitor.

## Archivos del proyecto

| Archivo | Descripción |
|---|---|
| `DeepLang.g4` | La gramática formal — define la sintaxis del lenguaje |
| `lexer.py` | Lee el código fuente y lo convierte en tokens |
| `parser.py` | Toma los tokens y construye el árbol de sintaxis |
| `EvalVisitor.py` | Recorre el árbol y ejecuta las operaciones |
| `DeepLang.py` | Punto de entrada — modo interactivo (REPL) |

## Cómo ejecutar

```bash
python3 DeepLang.py
```

El REPL muestra `dl>` cuando espera una instrucción nueva y `...` cuando está dentro de un bloque sin cerrar (`if`, `while`, `fun`).

---

## Sintaxis

### Aritmética

```
5 + 3
10 - 4
6 * 7
20 / 4
(2 ^ 3)
(1 + 2) * 3
```

La división es entera. La potencia se escribe con `^`. Los paréntesis cambian la precedencia normal (`^` antes que `*` y `/`, y luego `+` y `-`).

---

### Variables

```
a = 10
b = a + 5
```

Si usas una variable que no existe, vale `0`.

---

### Imprimir

```
print(x)
print(a + b)
```

Escribir una expresión sola **no** la imprime. Solo `print()` muestra valores en pantalla.

---

### Condicionales

```
if a > b
  print(a)
else
  print(b)
end
```

El `else` es opcional:

```
if x == 10
  print(x)
end
```

Operadores de comparación disponibles: `==`, `!=`, `<`, `>`, `<=`, `>=`.

---

### Ciclos

```
i = 1
while i <= 5
  print(i)
  i = i + 1
end
```

El bloque se repite mientras la condición sea verdadera.

---

### Arreglos

```
lista = [10, 20, 30]   
lista[0] = 99          
print(lista[0])        
```

- Los índices empiezan en `0`.
- Si el índice está fuera del rango, da error.

Recorrer un arreglo:

```
i = 0
while i < 3
  print(lista[i])
  i = i + 1
end
```

---

### Funciones y recursividad

```
fun factorial(n)
  if n == 0
    return 1
  end
  return n * factorial(n - 1)
end

print(factorial(5))
```

- Se declaran con `fun nombre(params)` y se cierran con `end`.
- Se retorna un valor con `return`.
- Soportan recursividad.
- Cada llamada tiene su propio ámbito de variables — los parámetros no pisan las variables externas.

---

### Funciones trigonométricas integradas

También puedes usar funciones trigonométricas sin definirlas:

```
x = 0
print(sen(x))
print(cos(x))
print(tan(x))
print(cosecante(x))
print(secante(x))
print(cotangente(x))
print(modulo(10, 3))
print(raiz(9))
```

- `sen`, `cos`, `tan`, `cosecante`, `secante` y `cotangente` reciben **1 argumento**.
- `sin` también está disponible como alias de `sen`.
- También hay alias cortos: `csc`, `sec`, `cot` y `ctg`.
- `modulo(a, b)` (alias `mod(a, b)`) calcula el residuo de `a % b`.
- `raiz(x)` calcula raíz cuadrada de `x`.
- El argumento está en **radianes**.
- Están implementadas desde cero en el intérprete (series de Taylor).

### Librería de lectura de archivos

DeepLang incluye funciones integradas para leer archivos de texto sin usar una librería `sys`:

```
contenido = leerarchivo("datos.txt")
print(contenido)

lineas = leerlineas("datos.txt")
print(lineas[0])
```

- `leerarchivo(ruta)` devuelve todo el contenido como texto (`string`).
- `leerlineas(ruta)` devuelve un arreglo con cada línea del archivo.
- Si la ruta no existe o no se puede leer, el intérprete muestra un error claro.

---

### Potencias

Puedes elevar usando `^`:

```
print(2 ^ 8)
print(9 ^ 2)
```

- `x ^ y` requiere que `y` sea entero.

---

## Ejemplo completo

```
fun fibonacci(n)
  if n == 0
    return 0
  end
  if n == 1
    return 1
  end
  return fibonacci(n - 1) + fibonacci(n - 2)
end

nums = [0, 1, 2, 3, 4, 5, 6, 7]
i = 0
while i < 8
  print(fibonacci(nums[i]))
  i = i + 1
end
```

Salida:
```
0
1
1
2
3
5
8
13
```

---

## Flujo del programa

```
Código fuente
      ↓
  lexer.py       →  convierte texto en tokens
      ↓
  parser.py      →  construye el árbol de sintaxis
      ↓
  EvalVisitor.py →  ejecuta las operaciones
      ↓
  Resultado en pantalla
```
