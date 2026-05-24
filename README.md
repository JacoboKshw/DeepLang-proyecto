# DeepLang

DeepLang es un lenguaje interpretado para computación científica, visualización de datos y machine learning.

## Ejecutar

```bash
python3 DeepLang.py              # REPL interactivo
python3 DeepLang.py script.dl    # Ejecutar archivo
```

## Sintaxis Básica

### Variables y aritmética
```deepdl
a = 10
b = 20
print(a + b)      # 30
print(2 ^ 3)      # 8 (potencia)
```

### Arreglos
```deepdl
lista = [1, 2, 3, 4]
print(lista[0])        # 1
print(longitud(lista)) # 4
lista[0] = 99
```

### If / While / For
```deepdl
if a > b
  print("Mayor")
end

while a < 100
  print(a)
  a = a + 1
end

for i = 1 to 5
  print(i)
end
```

### Funciones
```deepdl
fun suma(x, y)
  return x + y
end

print(suma(3, 4))  # 7
```

## Librerías Integradas

### Trigonometría (radianes)
```deepdl
sen(x), cos(x), tan(x)
cosecante(x), secante(x), cotangente(x)
```

### Matemática
```deepdl
abs(x)              # Valor absoluto
raiz(x)             # Raíz cuadrada
log(x), log10(x)    # Logaritmos
exp(x)              # Exponencial
redondear(x, d)     # Redondear
piso(x), techo(x)   # Piso y techo
modulo(a, b)        # Módulo
```

## Gráficas ASCII

```deepdl
grafica_barras(datos, n, ancho)
grafica_barras_v(datos, n, alto)
grafica_linea(datos, n, ancho, alto)
grafica_dispersion(X, Y, n, w, h)
histograma(datos, n, bins, ancho)
grafica_pastel(datos, n)
grafica_funcion(Y, n, xmin, xmax, ancho, alto)
```

Ejemplo:
```deepdl
datos = [5, 10, 15, 20, 25]
grafica_barras(datos, 5, 30)
```

## Matrices

```deepdl
M = mat_ceros(2, 3)              # Matriz de ceros
I = mat_identidad(3)             # Identidad
A = mat_suma(M1, M2, filas, cols)
B = mat_mul(A, B, m, n, p)       # Multiplicación
det = mat_det(A, n)              # Determinante
mat_imprimir(A, filas, cols)
```

## Machine Learning

### Normalización
```deepdl
m = media(datos, n)
std = desviacion(datos, n)
norm = normalizar_zscore(datos, n)
```

### Regresión Lineal
```deepdl
params = regresion_lineal(X, Y, n, lr, epochs)  # [w, b]
y_pred = rl_predecir(params, X, n)
```

### Red Neuronal (MLP)
```deepdl
ml_semilla(42)
red = mlp_init(2, 4, 1)              # entrada=2, oculta=4, salida=1
red = mlp_entrenar(red, X, Y, n, lr, epochs)
y_pred = mlp_predecir(red, X, n)
```

### K-Means
```deepdl
etiquetas = kmeans(X, n, m, k, max_iter)
kmeans_resumen(etiquetas, n, k)
```

### Otros algoritmos
```deepdl
# Regresión logística
params = regresion_logistica(X, Y, n, lr, epochs)

# Perceptrón
W = perceptron(X, Y, n, m, lr, epochs)

# Métricas
mse = mse(y_real, y_pred, n)
mae = mae(y_real, y_pred, n)
r2 = r_cuadrado(y_real, y_pred, n)
acc = accuracy(y_real, y_pred, n)
```

## Archivos

### Lectura de texto
```deepdl
contenido = leerarchivo("datos.txt")
lineas = leerlineas("datos.txt")
```

### Escritura de texto
```deepdl
escribirarchivo("salida.txt", "Contenido")
agregararchivo("salida.txt", "Más contenido")
```

### CSV
```deepdl
tabla = leercsv("datos.csv")              # Con encabezado
datos = leercsv_datos("datos.csv")        # Sin encabezado
columna = leercsv_columna("datos.csv", 0)
escribircsv("salida.csv", tabla)
```

## Ejemplo Completo

```deepdl
# Fibonacci
fun fib(n)
  if n <= 1
    return n
  end
  return fib(n-1) + fib(n-2)
end

i = 0
while i < 10
  print(fib(i))
  i = i + 1
end
```

## Archivos del Proyecto

- `DeepLang.py` — Intérprete (REPL)
- `lexer.py` — Tokenizador
- `parser.py` — Analizador sintáctico
- `EvalVisitor.py` — Evaluador
- `deeplang_filelib.py` — Archivos y CSV
- `deeplang_graficaslib.py` — Gráficas
- `deeplang_matriceslib.py` — Matrices
- `deeplang_mllib.py` — Machine Learning

## Notas

- Los índices empiezan en 0
- Las funciones se pasan por valor (reasignar si modificas)
- Variables no asignadas valen 0
- Trigonometría en radianes
- Sin dependencias externas (Python puro)

**Autores:** Jacobo Mondragón, Julián Gómez
