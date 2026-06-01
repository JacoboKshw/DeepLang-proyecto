# DeepLang - Lenguaje Interpretado con Machine Learning

## Tabla de Contenidos
1. [Sintaxis Básica](#sintaxis-básica)
2. [Machine Learning](#machine-learning)
   - [Activaciones](#activaciones)
   - [Normalización](#normalización)
   - [Métricas](#métricas)
   - [Regresión Lineal](#regresión-lineal)
   - [Regresión Logística](#regresión-logística)
   - [Redes Neuronales](#redes-neuronales-mlp)
   - [K-Means](#k-means)
3. [Gráficas](#gráficas-ascii)

---

# Sintaxis Básica

```deepdl
# Variables y operadores
a = 10
b = 20
c = a + b

# Arrays
lista = [1, 2, 3, 4, 5]
lista[0] = 99

# Condicionales
if a > 5
  print("Mayor")
else
  print("Menor")
end

# Bucles
while i < 10
  i = i + 1
end

for j = 0 to 5
  print(j)
end

# Funciones
fun suma(x, y)
  return x + y
end

resultado = suma(3, 5)

# Operadores
poder = 2 ^ 3           # Potencia (8)
division = 10 / 3       # División (3.33)
resto = modulo(10, 3)   # Resto (1)
raiz = raiz(16)         # Raíz cuadrada (4)
```

---

# Machine Learning

## Activaciones

### sig(x) - Sigmoide
```deepdl
resultado = sig(0)       # 0.5
resultado = sig(2)       # 0.88
resultado = sig(-2)      # 0.12
```
**Parámetros:**
| Parámetro | Tipo | Rango | Descripción |
|-----------|------|-------|-------------|
| `x` | número | (-∞, ∞) | Valor de entrada |

**Retorna:** Número en rango [0, 1]

**Fórmula:** 1 / (1 + e^-x)

**Uso:** Regresión logística, capas de salida en redes neuronales

---

### sig_deriv(s) - Derivada de Sigmoide
```deepdl
s = sig(1)
deriv = sig_deriv(s)     # 0.196
```
**Parámetros:**
| Parámetro | Tipo | Rango | Descripción |
|-----------|------|-------|-------------|
| `s` | número | [0, 1] | Valor de sigmoide |

**Retorna:** Número

**Fórmula:** s * (1 - s)

**Uso:** Backpropagation en redes neuronales

---

### relu(x) - ReLU (Rectified Linear Unit)
```deepdl
print(relu(-2))          # 0
print(relu(0))           # 0
print(relu(3))           # 3
```
**Parámetros:**
| Parámetro | Tipo | Rango | Descripción |
|-----------|------|-------|-------------|
| `x` | número | (-∞, ∞) | Valor de entrada |

**Retorna:** Número en rango [0, ∞)

**Fórmula:** max(0, x)

**Uso:** Capas ocultas en redes profundas (muy popular)

---

### relu_deriv(x) - Derivada de ReLU
```deepdl
print(relu_deriv(-2))    # 0
print(relu_deriv(0))     # 0
print(relu_deriv(3))     # 1
```
**Parámetros:**
| Parámetro | Tipo | Rango | Descripción |
|-----------|------|-------|-------------|
| `x` | número | (-∞, ∞) | Valor de entrada |

**Retorna:** 0 o 1

**Fórmula:** 1 si x > 0, sino 0

**Uso:** Backpropagation en redes con ReLU

---

### tanh_dl(x) - Tangente Hiperbólica
```deepdl
print(tanh_dl(0))        # 0
print(tanh_dl(1))        # 0.761
print(tanh_dl(-1))       # -0.761
```
**Parámetros:**
| Parámetro | Tipo | Rango | Descripción |
|-----------|------|-------|-------------|
| `x` | número | (-∞, ∞) | Valor de entrada |

**Retorna:** Número en rango [-1, 1]

**Fórmula:** (e^x - e^-x) / (e^x + e^-x)

**Uso:** Redes recurrentes, cuando necesitas salidas negativas

---

### tanh_deriv(t) - Derivada de Tanh
```deepdl
t = tanh_dl(0.5)
print(tanh_deriv(t))     # 0.787
```
**Parámetros:**
| Parámetro | Tipo | Rango | Descripción |
|-----------|------|-------|-------------|
| `t` | número | [-1, 1] | Valor de tanh |

**Retorna:** Número

**Fórmula:** 1 - t^2

**Uso:** Backpropagation

---

## Normalización

### media(datos, n) - Promedio
```deepdl
datos = [10, 20, 30, 40, 50]
n = 5
prom = media(datos, n)   # 30
```
**Parámetros:**
| Parámetro | Tipo | Rango | Descripción |
|-----------|------|-------|-------------|
| `datos` | arreglo | - | Array de números |
| `n` | int | 1-10000 | Cantidad de elementos |

**Retorna:** Número (promedio)

**Fórmula:** sum(datos) / n

**Uso:** Preprocesamiento, análisis exploratorio

---

### desviacion(datos, n) - Desviación Estándar
```deepdl
datos = [10, 20, 30, 40, 50]
n = 5
std = desviacion(datos, n)  # 15.81
```
**Parámetros:**
| Parámetro | Tipo | Rango | Descripción |
|-----------|------|-------|-------------|
| `datos` | arreglo | - | Array de números |
| `n` | int | 1-10000 | Cantidad de elementos |

**Retorna:** Número (desviación estándar)

**Fórmula:** sqrt(sum((x - media)^2) / n)

**Uso:** Normalización Z-score, análisis de dispersión

---

### normalizar_zscore(datos, n) - Normalización Z-Score
```deepdl
datos = [10, 20, 30, 40, 50]
n = 5
norm = normalizar_zscore(datos, n)
# [-1.414, -0.707, 0, 0.707, 1.414]
```
**Parámetros:**
| Parámetro | Tipo | Rango | Descripción |
|-----------|------|-------|-------------|
| `datos` | arreglo | - | Array de números |
| `n` | int | 1-10000 | Cantidad de elementos |

**Retorna:** Arreglo normalizado

**Fórmula:** (x - media) / desviación_estándar

**Resultado:** Media=0, Desv. Est.=1

**CRÍTICO:** Normalizar SIEMPRE antes de regresión lineal y redes neuronales

**Uso:** Preprocesamiento de datos

---

### normalizar_minmax(datos, n) - Normalización Min-Max
```deepdl
datos = [10, 20, 30, 40, 50]
n = 5
norm = normalizar_minmax(datos, n)
# [0, 0.25, 0.5, 0.75, 1]
```
**Parámetros:**
| Parámetro | Tipo | Rango | Descripción |
|-----------|------|-------|-------------|
| `datos` | arreglo | - | Array de números |
| `n` | int | 1-10000 | Cantidad de elementos |

**Retorna:** Arreglo normalizado en [0, 1]

**Fórmula:** (x - min) / (max - min)

**Uso:** Normalización al rango [0, 1], imágenes, múltiples escalas

---

## Métricas

### mse(y_real, y_pred, n) - Error Cuadrático Medio
```deepdl
y_real = [1, 2, 3, 4, 5]
y_pred = [1.1, 2.2, 2.9, 4.1, 4.8]
n = 5
error = mse(y_real, y_pred, n)  # 0.024
```
**Parámetros:**
| Parámetro | Tipo | Rango | Descripción |
|-----------|------|-------|-------------|
| `y_real` | arreglo | (-∞, ∞) | Valores reales |
| `y_pred` | arreglo | (-∞, ∞) | Predicciones |
| `n` | int | 1-10000 | Cantidad de muestras |

**Retorna:** Número >= 0

**Fórmula:** sum((y_real - y_pred)^2) / n

**Interpretación:**
- MSE = 0 → Predicción perfecta
- MSE bajo → Buen modelo
- MSE alto → Mal modelo
- Penaliza mucho los errores grandes

**Uso:** Regresión lineal, evaluación de precisión

---

### mae(y_real, y_pred, n) - Error Absoluto Medio
```deepdl
y_real = [1, 2, 3, 4, 5]
y_pred = [1.1, 2.2, 2.9, 4.1, 4.8]
n = 5
error = mae(y_real, y_pred, n)  # 0.16
```
**Parámetros:**
| Parámetro | Tipo | Rango | Descripción |
|-----------|------|-------|-------------|
| `y_real` | arreglo | (-∞, ∞) | Valores reales |
| `y_pred` | arreglo | (-∞, ∞) | Predicciones |
| `n` | int | 1-10000 | Cantidad de muestras |

**Retorna:** Número >= 0

**Fórmula:** sum(|y_real - y_pred|) / n

**Interpretación:**
- MAE = 0 → Predicción perfecta
- MAE en mismas unidades que y
- Robusto a outliers

**Uso:** Métrica robusta, datos con valores extremos

---

### r_cuadrado(y_real, y_pred, n) - Coeficiente R²
```deepdl
y_real = [1, 2, 3, 4, 5]
y_pred = [1.1, 2.2, 2.9, 4.1, 4.8]
n = 5
r2 = r_cuadrado(y_real, y_pred, n)  # 0.998
```
**Parámetros:**
| Parámetro | Tipo | Rango | Descripción |
|-----------|------|-------|-------------|
| `y_real` | arreglo | (-∞, ∞) | Valores reales |
| `y_pred` | arreglo | (-∞, ∞) | Predicciones |
| `n` | int | 1-10000 | Cantidad de muestras |

**Retorna:** Número

**Rango:** (-∞, 1]

**Fórmula:** 1 - (SS_res / SS_tot)

**Interpretación:**
- R² = 1.0 → Predicción perfecta
- R² = 0.5 → Explica 50% de la varianza
- R² = 0.0 → Tan bueno como predecir la media
- R² < 0 → Peor que predecir la media

**Uso:** Evaluar regresión, reportes, comparar modelos

---

### accuracy(y_real, y_pred, n) - Exactitud
```deepdl
y_real = [1, 0, 1, 1, 0, 1, 0, 0]
y_pred = [0.9, 0.1, 0.8, 1.0, 0.2, 0.6, 0.3, 0.1]
n = 8
acc = accuracy(y_real, y_pred, n)  # 1.0
```
**Parámetros:**
| Parámetro | Tipo | Rango | Descripción |
|-----------|------|-------|-------------|
| `y_real` | arreglo | {0, 1} | Etiquetas verdaderas |
| `y_pred` | arreglo | [0, 1] | Predicciones (probabilidades) |
| `n` | int | 1-10000 | Cantidad de muestras |

**Retorna:** Número

**Rango:** [0, 1]

**Fórmula:** (correctos / total) * 100

**Nota:** Usa threshold 0.5
- y_pred >= 0.5 → predice 1
- y_pred < 0.5 → predice 0

**Uso:** Clasificación binaria, evaluación

---

## Regresión Lineal

### regresion_lineal(X, Y, n, lr, epochs) - Entrenar Modelo Lineal
```deepdl
X = [1, 2, 3, 4, 5]
Y = [2.1, 3.9, 6.2, 7.8, 10.1]
n = 5

# Normalizar SIEMPRE
X_norm = normalizar_zscore(X, n)
Y_norm = normalizar_zscore(Y, n)

params = regresion_lineal(X_norm, Y_norm, n, 0.01, 500)
# params = [w, b]
```
**Parámetros:**
| Parámetro | Tipo | Rango | Recomendado | Descripción |
|-----------|------|-------|-------------|-------------|
| `X` | arreglo | (-∞, ∞) | - | Valores independientes |
| `Y` | arreglo | (-∞, ∞) | - | Valores dependientes (targets) |
| `n` | int | 1-10000 | - | Cantidad de muestras |
| `lr` | número | 0.0001-1.0 | 0.01 | Learning rate (velocidad) |
| `epochs` | int | 1-10000 | 500 | Iteraciones de entrenamiento |

**Learning Rate (`lr`) - Guía:**
| Valor | Velocidad | Convergencia | Uso |
|-------|-----------|--------------|-----|
| 0.001 | Muy lento | Garantizada | Datos sensibles |
| 0.01 | Lento | Garantizada | Balance típico |
| 0.1 | Rápido | Puede fallar | Datos simples |
| 1.0+ | Muy rápido | Diverge | Evitar |

**Epochs (`epochs`) - Guía:**
| Valor | Complejidad | Uso |
|-------|-----------|-----|
| 100 | Baja | Datos muy simples |
| 500 | Media | Balance típico |
| 1000 | Alta | Datos complejos o lr bajo |
| 5000+ | Muy alta | Debugging |

**Retorna:** [w, b] (arreglo con 2 elementos)
- `w` = peso (pendiente)
- `b` = sesgo (intercepto)

**Fórmula:** y = w*x + b

**IMPORTANTE:** 
- Normalizar X e Y con `normalizar_zscore()` antes de usar
- Si diverge (pérdida aumenta), reducir `lr`
- Si converge muy rápido, aumentar `epochs`

**Uso:** Predicción lineal simple, análisis de correlación

---

### rl_predecir(params, X, n) - Predicción Lineal
```deepdl
params = [2.0, 0.05]  # [w, b] del modelo entrenado
X = [1, 2, 3, 4, 5]
n = 5

y_pred = rl_predecir(params, X, n)
# [2.05, 4.05, 6.05, 8.05, 10.05]
```
**Parámetros:**
| Parámetro | Tipo | Rango | Descripción |
|-----------|------|-------|-------------|
| `params` | arreglo | - | [w, b] retornados por `regresion_lineal()` |
| `X` | arreglo | (-∞, ∞) | Valores para predecir |
| `n` | int | 1-10000 | Cantidad de valores |

**Retorna:** Arreglo de predicciones

**Fórmula:** y_pred = w*X + b

**Nota:** Si X fue normalizado al entrenar, denormalizar predicciones

**Uso:** Predicción después de entrenar modelo

---

## Regresión Lineal Múltiple

### regresion_lineal_multi(X_plano, Y, n, m, lr, epochs)
```deepdl
# Ejemplo: Predecir ingreso por edad + años de escuela
datos = leerxlsx_datos("data.xlsx")
n = longitud(datos)
m = 2  # 2 features

# Crear matriz plana
X_plano = []
Y = []

i = 0
while i < n
  edad = flotante(datos[i][1])
  anios = flotante(datos[i][5])
  ingreso = flotante(datos[i][3])
  
  X_plano = X_plano + [edad, anios]
  Y = Y + [ingreso]
  
  i = i + 1
end

# Normalizar
X_norm = normalizar_zscore(X_plano, n*m)
Y_norm = normalizar_zscore(Y, n)

# Entrenar
W = regresion_lineal_multi(X_norm, Y_norm, n, m, 0.01, 500)
```
**Parámetros:**
| Parámetro | Tipo | Rango | Recomendado | Descripción |
|-----------|------|-------|-------------|-------------|
| `X_plano` | arreglo | (-∞, ∞) | - | Matriz n×m en formato plano |
| `Y` | arreglo | (-∞, ∞) | - | Targets (arreglo de tamaño n) |
| `n` | int | 1-10000 | - | Cantidad de muestras |
| `m` | int | 1-100 | 2-10 | Cantidad de features |
| `lr` | número | 0.0001-1.0 | 0.01 | Learning rate |
| `epochs` | int | 1-10000 | 500 | Iteraciones |

**Formato Plano (row-major):**
```
Matriz lógica (3×2):     Formato plano:
[1, 2]                   [1, 2, 3, 4, 5, 6]
[3, 4]
[5, 6]

Para acceder X[i][j] en formato plano:
X_plano[i*m + j]  donde m=2

Ejemplo: Fila 1, columna 1 (valor 4):
X_plano[1*2 + 1] = X_plano[3] = 4
```

**Retorna:** [w0, w1, w2, ..., wm] (m+1 elementos)
- w0 = sesgo (bias)
- w1...wm = pesos de cada feature

**Fórmula:** y = w0 + w1*x1 + w2*x2 + ... + wm*xm

**Uso:** Predicción con múltiples features

**Nota:** Normalizar datos antes de usar

---

### rl_multi_predecir(W, X_plano, n, m)
```deepdl
W = [100, 0.5, 10]  # [w0, w1, w2]
X_plano = [25, 10, 35, 15, 45, 12]  # 3 muestras, 2 features
n = 3
m = 2

y_pred = rl_multi_predecir(W, X_plano, n, m)
# Predicciones para 3 muestras
```
**Parámetros:**
| Parámetro | Tipo | Rango | Descripción |
|-----------|------|-------|-------------|
| `W` | arreglo | - | Parámetros [w0, w1, ..., wm] |
| `X_plano` | arreglo | (-∞, ∞) | Matriz en formato plano |
| `n` | int | 1-10000 | Cantidad de muestras |
| `m` | int | 1-100 | Cantidad de features |

**Retorna:** Arreglo de n predicciones

**Uso:** Predicción múltiple

---

## Regresión Logística

### regresion_logistica(X, Y, n, lr, epochs)
```deepdl
X = [0, 1, 2, 3, 4, 5]
Y = [0, 0, 0, 1, 1, 1]  # Clasificación binaria
n = 6

# Normalizar
X_norm = normalizar_zscore(X, n)
Y_norm = normalizar_zscore(Y, n)

params = regresion_logistica(X_norm, Y_norm, n, 0.1, 200)
```
**Parámetros:**
| Parámetro | Tipo | Rango | Recomendado | Descripción |
|-----------|------|-------|-------------|-------------|
| `X` | arreglo | (-∞, ∞) | - | Valores independientes |
| `Y` | arreglo | {0, 1} | - | Etiquetas binarias |
| `n` | int | 1-10000 | - | Cantidad de muestras |
| `lr` | número | 0.001-1.0 | 0.1 | Learning rate |
| `epochs` | int | 1-5000 | 200-500 | Iteraciones |

**Retorna:** [w, b]

**Fórmula:** P(y=1|x) = sigmoide(w*x + b)

**Uso:** Clasificación binaria

---

### log_predecir(params, X, n)
```deepdl
params = [1.5, 0.2]
X = [0, 1, 2, 3, 4, 5]
n = 6

probs = log_predecir(params, X, n)
# [0.550, 0.656, 0.747, 0.820, 0.873, 0.911]
```
**Parámetros:**
| Parámetro | Tipo | Rango | Descripción |
|-----------|------|-------|-------------|
| `params` | arreglo | - | [w, b] del modelo |
| `X` | arreglo | (-∞, ∞) | Valores para predecir |
| `n` | int | 1-10000 | Cantidad de valores |

**Retorna:** Arreglo de probabilidades [0, 1]

**Uso:** Predicción probabilística

---

### regresion_logistica_multi(X_plano, Y, n, m, lr, epochs)
```deepdl
X_plano = [25, 10, 35, 15, 45, 12]  # 3 muestras, 2 features
Y = [0, 1, 1]
n = 3
m = 2

W = regresion_logistica_multi(X_plano, Y, n, m, 0.01, 300)
```
**Parámetros:**
| Parámetro | Tipo | Rango | Recomendado | Descripción |
|-----------|------|-------|-------------|-------------|
| `X_plano` | arreglo | (-∞, ∞) | - | Matriz n×m en formato plano |
| `Y` | arreglo | {0, 1} | - | Etiquetas binarias |
| `n` | int | 1-10000 | - | Cantidad de muestras |
| `m` | int | 1-100 | 1-20 | Cantidad de features |
| `lr` | número | 0.001-1.0 | 0.01 | Learning rate |
| `epochs` | int | 1-5000 | 300 | Iteraciones |

**Retorna:** [w0, w1, ..., wm]

**Uso:** Clasificación binaria multi-feature

---

### log_multi_predecir(W, X_plano, n, m)
```deepdl
W = [0.1, 0.05, 0.3]
X_plano = [25, 10, 35, 15, 45, 12]
n = 3
m = 2

probs = log_multi_predecir(W, X_plano, n, m)
```
**Parámetros:**
| Parámetro | Tipo | Rango | Descripción |
|-----------|------|-------|-------------|
| `W` | arreglo | - | Parámetros [w0, w1, ..., wm] |
| `X_plano` | arreglo | (-∞, ∞) | Matriz en formato plano |
| `n` | int | 1-10000 | Cantidad de muestras |
| `m` | int | 1-100 | Cantidad de features |

**Retorna:** Arreglo de probabilidades

**Uso:** Predicción probabilística multi-feature

---

## Perceptrón

### perceptron(X_plano, Y, n, m, lr, epochs)
```deepdl
X_plano = [0.25, 0.4, 0.35, 0.6, 0.45, 0.48]  # 3 muestras, 2 features
Y = [0, 1, 0]
n = 3
m = 2

W = perceptron(X_plano, Y, n, m, 0.5, 100)
```
**Parámetros:**
| Parámetro | Tipo | Rango | Recomendado | Descripción |
|-----------|------|-------|-------------|-------------|
| `X_plano` | arreglo | (-∞, ∞) | - | Matriz n×m en formato plano |
| `Y` | arreglo | {0, 1} | - | Etiquetas binarias |
| `n` | int | 1-10000 | - | Cantidad de muestras |
| `m` | int | 1-100 | 1-20 | Cantidad de features |
| `lr` | número | 0.01-2.0 | 0.5 | Learning rate |
| `epochs` | int | 1-1000 | 50-200 | Iteraciones |

**Retorna:** [w0, w1, ..., wm]

**Fórmula:** y = signo(w0 + w1*x1 + w2*x2 + ...)

**Uso:** Clasificación lineal simple

---

### perceptron_predecir(W, X_plano, n, m)
```deepdl
W = [0.1, 0.5, 0.3]
X_plano = [0.25, 0.4, 0.35, 0.6]  # 2 muestras
n = 2
m = 2

y_pred = perceptron_predecir(W, X_plano, n, m)
# [0, 1] etiquetas duras
```
**Parámetros:**
| Parámetro | Tipo | Rango | Descripción |
|-----------|------|-------|-------------|
| `W` | arreglo | - | Parámetros [w0, w1, ..., wm] |
| `X_plano` | arreglo | (-∞, ∞) | Matriz en formato plano |
| `n` | int | 1-10000 | Cantidad de muestras |
| `m` | int | 1-100 | Cantidad de features |

**Retorna:** Arreglo de etiquetas {0, 1}

**Uso:** Predicción perceptrón

---

## Redes Neuronales (MLP)

### ml_semilla(s) - Establecer Semilla Aleatoria
```deepdl
ml_semilla(42)
red = mlp_init(2, 8, 1)  # Mismos pesos cada vez
```
**Parámetros:**
| Parámetro | Tipo | Rango | Recomendado | Descripción |
|-----------|------|-------|-------------|-------------|
| `s` | int | 1-1000000 | 1-1000 | Semilla para reproducibilidad |

**Retorna:** -

**IMPORTANTE:** Llamar SIEMPRE antes de `mlp_init()` para reproducibilidad

**Uso:** Fijar seed del generador aleatorio

---

### mlp_init(n_in, n_hidden, n_out)
```deepdl
ml_semilla(42)
red = mlp_init(2, 8, 1)
# 2 inputs → 8 neuronas ocultas → 1 salida
```
**Parámetros:**
| Parámetro | Tipo | Rango | Recomendado | Descripción |
|-----------|------|-------|-------------|-------------|
| `n_in` | int | 1-100 | 1-20 | Neuronas de entrada (features) |
| `n_hidden` | int | 1-512 | 4-64 | Neuronas ocultas |
| `n_out` | int | 1-100 | 1-10 | Neuronas de salida |

**n_hidden - Guía:**
| Regla | n_hidden | Uso |
|------|----------|-----|
| Pocos datos | 2-8 | Evitar overfitting |
| Datos medios | 8-16 | Balance típico |
| Muchos datos | 16-64 | Más capacidad |
| Muy complejos | 64-512 | Problemas difíciles |

**Fórmula:** n_hidden ≈ 2 * n_in (como inicio)

**Retorna:** red (arreglo interno)

**Uso:** Crear red neuronal

---

### mlp_forward(red, x, n_in) - Forward Pass Manual
```deepdl
red = mlp_init(2, 4, 1)
x = [0.5, 0.3]
n_in = 2

salida = mlp_forward(red, x, n_in)
# [h0, h1, h2, h3, o0]
# h = neuronas ocultas, o = neuronas salida

prediccion = salida[4]  # Última: neuron de salida
```
**Parámetros:**
| Parámetro | Tipo | Rango | Descripción |
|-----------|------|-------|-------------|
| `red` | - | - | Red creada por `mlp_init()` |
| `x` | arreglo | (-∞, ∞) | Vector de inputs (tamaño n_in) |
| `n_in` | int | 1-100 | Cantidad de inputs |

**Retorna:** [h0...h_hidden, o0...o_out]

**Estructura del retorno:**
```
Para red: n_in=2, n_hidden=4, n_out=1
Retorna: [h0, h1, h2, h3, o0]
         └─────────┬─────────┘ └─┬─┘
      neuronas ocultas (4)  salida (1)
      
Para acceder salida:
salida = mlp_forward(red, x, n_in)
prediccion = salida[n_hidden]  // Índice donde empieza la salida
```

**Uso:** Forward pass manual, debugging

---

### mlp_entrenar(red, X_plano, Y_plano, n, lr, epochs)
```deepdl
ml_semilla(42)

# Preparar datos
datos = leerxlsx_datos("data.xlsx")
n = longitud(datos)

X_plano = []
Y = []

i = 0
while i < n
  edad = flotante(datos[i][1]) / 100.0      # Normalizar 0-1
  anios = flotante(datos[i][5]) / 25.0
  ingreso = flotante(datos[i][3]) / 100000.0
  
  X_plano = X_plano + [edad, anios]
  Y = Y + [ingreso]
  
  i = i + 1
end

# Crear y entrenar
red = mlp_init(2, 8, 1)
red = mlp_entrenar(red, X_plano, Y, n, 0.1, 200)

print("Red entrenada")
```
**Parámetros:**
| Parámetro | Tipo | Rango | Recomendado | Descripción |
|-----------|------|-------|-------------|-------------|
| `red` | - | - | - | Red de `mlp_init()` |
| `X_plano` | arreglo | (-∞, ∞) | - | Matriz n×n_in en formato plano |
| `Y_plano` | arreglo | (-∞, ∞) | - | Targets (tamaño n) |
| `n` | int | 1-10000 | - | Cantidad de muestras |
| `lr` | número | 0.0001-1.0 | 0.1 | Learning rate |
| `epochs` | int | 1-5000 | 200 | Iteraciones |

**Learning Rate (`lr`) - Guía:**
| Valor | Velocidad | Uso |
|-------|-----------|-----|
| 0.001 | Muy lento | Datos sensibles |
| 0.01 | Lento | Balance |
| 0.1 | Rápido | Típico |
| 0.5 | Muy rápido | Puede divergir |
| 1.0+ | Diverge | Evitar |

**Epochs (`epochs`) - Guía:**
| Valor | Complejidad | Uso |
|-------|-----------|-----|
| 100 | Baja | Problemas simples |
| 200-300 | Media | Balance típico |
| 500 | Alta | Problemas complejos |
| 1000+ | Muy alta | Problemas difíciles |

**Retorna:** red actualizada

**IMPORTANTE:**
- Normalizar datos antes de entrenar
- Si diverge (pérdida aumenta), reducir `lr`
- Si converge lentamente, aumentar `lr` o `epochs`
- Usar `mlp_semilla()` antes para reproducibilidad

**Uso:** Entrenamiento de red neuronal con backpropagation

---

### mlp_predecir(red, X_plano, n)
```deepdl
red_entrenada = ...  # red ya entrenada
X_test = [0.3, 0.4, 0.5, 0.6]  # 2 muestras, 2 features
n = 2

y_pred = mlp_predecir(red_entrenada, X_test, n)
# [0.23, 0.67] predicciones para 2 muestras
```
**Parámetros:**
| Parámetro | Tipo | Rango | Descripción |
|-----------|------|-------|-------------|
| `red` | - | - | Red entrenada |
| `X_plano` | arreglo | (-∞, ∞) | Matriz n×n_in en formato plano |
| `n` | int | 1-10000 | Cantidad de muestras |

**Retorna:** Arreglo de n predicciones

**Uso:** Predicción con red entrenada

---

## K-Means

### kmeans(X_plano, n, m, k, max_iter)
```deepdl
# Agrupar ingresos en 3 clusters
datos = leerxlsx_datos("data.xlsx")
n = longitud(datos)
m = 2  # edad + ingreso

X_plano = []
i = 0
while i < n
  edad = flotante(datos[i][1]) / 100.0
  ingreso = flotante(datos[i][3]) / 100000.0
  X_plano = X_plano + [edad, ingreso]
  i = i + 1
end

k = 3
etiquetas = kmeans(X_plano, n, m, k, 100)

kmeans_resumen(etiquetas, n, k)
```
**Parámetros:**
| Parámetro | Tipo | Rango | Recomendado | Descripción |
|-----------|------|-------|-------------|-------------|
| `X_plano` | arreglo | (-∞, ∞) | - | Datos n×m en formato plano |
| `n` | int | 1-10000 | - | Cantidad de muestras |
| `m` | int | 1-100 | 1-10 | Cantidad de features |
| `k` | int | 1-100 | 2-10 | Número de clusters |
| `max_iter` | int | 1-1000 | 50-200 | Iteraciones máximas |

**k - Guía (Método del Codo):**
| k | Uso |
|---|-----|
| 2-3 | Ver 2-3 grupos principales |
| 5 | Balance típico |
| 10+ | Búsqueda fina de grupos |

**Algoritmo:**
1. Inicializar k centroides (primeros k puntos)
2. E-step: Asignar cada punto al centroide más cercano
3. M-step: Recalcular centroides como promedio del grupo
4. Repetir hasta convergencia o max_iter

**Retorna:** Arreglo de etiquetas [0..k-1] para cada muestra

**Uso:** Agrupamiento no supervisado

---

### kmeans_resumen(etiquetas, n, k)
```deepdl
etiquetas = kmeans(X_plano, 10280, 2, 3, 100)

kmeans_resumen(etiquetas, 10280, 3)

# Output:
# --- Resumen K-Means ---
# Cluster 0: 3427 puntos
# Cluster 1: 3856 puntos
# Cluster 2: 3097 puntos
```
**Parámetros:**
| Parámetro | Tipo | Rango | Descripción |
|-----------|------|-------|-------------|
| `etiquetas` | arreglo | [0..k-1] | Etiquetas de `kmeans()` |
| `n` | int | 1-10000 | Cantidad total de muestras |
| `k` | int | 1-100 | Número de clusters |

**Retorna:** - (imprime resumen)

**Uso:** Resumen de clusters

---

# Gráficas ASCII

## grafica_barras(datos, n, ancho)
```deepdl
datos = [5, 10, 15, 20, 25]
n = 5

grafica_barras(datos, n, 30)

# Output:
# ■■■■■■           (5)
# ■■■■■■■■■■       (10)
# ■■■■■■■■■■■■■■■  (15)
# ■■■■■■■■■■■■■■■■■■■■  (20)
# ■■■■■■■■■■■■■■■■■■■■■■■■■  (25)
```
**Parámetros:**
| Parámetro | Tipo | Rango | Recomendado | Descripción |
|-----------|------|-------|-------------|-------------|
| `datos` | arreglo | [0, ∞) | - | Valores para graficar |
| `n` | int | 1-100 | - | Cantidad de valores |
| `ancho` | int | 10-100 | 20-50 | Ancho máximo en caracteres |

**Retorna:** 0

**Uso:** Gráfica de barras horizontales

---

## grafica_barras_v(datos, n, alto)
```deepdl
datos = [5, 10, 15, 20, 25]
n = 5

grafica_barras_v(datos, n, 20)

# Output (columnas verticales):
#       ■
#       ■
#     ■ ■
#     ■ ■
#   ■ ■ ■
#   ■ ■ ■  
# ■ ■ ■ ■ ■
```
**Parámetros:**
| Parámetro | Tipo | Rango | Recomendado | Descripción |
|-----------|------|-------|-------------|-------------|
| `datos` | arreglo | [0, ∞) | - | Valores para graficar |
| `n` | int | 1-100 | - | Cantidad de valores |
| `alto` | int | 5-50 | 10-30 | Altura en filas |

**Retorna:** 0

**Uso:** Gráfica de barras verticales

---

## grafica_linea(datos, n, ancho, alto)
```deepdl
datos = [5, 10, 8, 15, 12, 20, 18]
n = 7

grafica_linea(datos, n, 40, 15)

# Output:
# 20 |               ■       
# 18 |              /   \
# 15 |          ■  /     ■
# 12 |            /       \
# 10 |    ■      /         \
#  8 |   / \    /
#  5 | ■/   \  /
```
**Parámetros:**
| Parámetro | Tipo | Rango | Recomendado | Descripción |
|-----------|------|-------|-------------|-------------|
| `datos` | arreglo | (-∞, ∞) | - | Valores Y |
| `n` | int | 1-100 | - | Cantidad de puntos |
| `ancho` | int | 20-100 | 30-80 | Ancho en columnas |
| `alto` | int | 5-50 | 10-30 | Altura en filas |

**Retorna:** 0

**Uso:** Gráfica de línea (serie temporal)

---

## grafica_dispersion(X, Y, n, w, h)
```deepdl
X = [1, 2, 3, 4, 5, 6]
Y = [2, 4, 6, 8, 10, 12]
n = 6

grafica_dispersion(X, Y, n, 30, 15)

# Output:
# 12 |              ○
# 10 |           ○
#  8 |        ○
#  6 |     ○
#  4 |  ○
#  2 |○
```
**Parámetros:**
| Parámetro | Tipo | Rango | Recomendado | Descripción |
|-----------|------|-------|-------------|-------------|
| `X` | arreglo | (-∞, ∞) | - | Valores X |
| `Y` | arreglo | (-∞, ∞) | - | Valores Y |
| `n` | int | 1-100 | - | Cantidad de puntos |
| `w` | int | 20-100 | 30-80 | Ancho en columnas |
| `h` | int | 5-50 | 10-30 | Altura en filas |

**Retorna:** 0

**Uso:** Diagrama de dispersión (scatter plot)

---

## histograma(datos, n, bins, ancho)
```deepdl
datos = [5, 10, 8, 15, 12, 20, 18, 25, 22, 30]
n = 10

histograma(datos, n, 5, 30)

# Output (5 bins):
# [0-6]    | ■         (1)
# [6-12]   | ■■■       (3)
# [12-18]  | ■■        (2)
# [18-24]  | ■■        (2)
# [24-30]  | ■■        (2)
```
**Parámetros:**
| Parámetro | Tipo | Rango | Recomendado | Descripción |
|-----------|------|-------|-------------|-------------|
| `datos` | arreglo | (-∞, ∞) | - | Valores para agrupar |
| `n` | int | 1-100 | - | Cantidad de valores |
| `bins` | int | 2-50 | 5-20 | Número de bins (intervalos) |
| `ancho` | int | 10-100 | 20-50 | Ancho máximo en caracteres |

**Cálculo de bins automático:**
```
Regla de Sturges: bins ≈ 1 + log2(n)

n=10  → bins ≈ 5
n=100 → bins ≈ 8
n=1000 → bins ≈ 11
```

**Retorna:** 0

**Uso:** Distribución de valores, análisis de frecuencias

---

## grafica_pastel(datos, n)
```deepdl
datos = [30, 25, 20, 25]
n = 4

grafica_pastel(datos, n)

# Output:
# Pastel ASCII:
# Sector 0: 30 puntos (30.0%)  ■■■■■■■■■■
# Sector 1: 25 puntos (25.0%)  ■■■■■■■■
# Sector 2: 20 puntos (20.0%)  ■■■■■■
# Sector 3: 25 puntos (25.0%)  ■■■■■■■■
```
**Parámetros:**
| Parámetro | Tipo | Rango | Recomendado | Descripción |
|-----------|------|-------|-------------|-------------|
| `datos` | arreglo | [0, ∞) | - | Valores/proporciones |
| `n` | int | 2-20 | 3-8 | Cantidad de sectores |

**Retorna:** 0

**Uso:** Proporciones, porcentajes

---

## grafica_funcion(Y, n, xmin, xmax, ancho, alto)
```deepdl
# Graficar función y = x^2 de -10 a 10
Y = [100, 81, 64, 49, 36, 25, 16, 9, 4, 1, 0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
n = 21

grafica_funcion(Y, n, -10, 10, 40, 15)

# Output:
# 100 |■               ■
#  80 | ■             ■
#  60 |  ■           ■
#  40 |   ■         ■
#  20 |     ■       ■
#   0 |      ■     ■
#      -10  -5  0  5  10
```
**Parámetros:**
| Parámetro | Tipo | Rango | Recomendado | Descripción |
|-----------|------|-------|-------------|-------------|
| `Y` | arreglo | (-∞, ∞) | - | Valores Y uniformes |
| `n` | int | 2-100 | - | Cantidad de puntos |
| `xmin` | número | (-∞, ∞) | - | Etiqueta X mínima (solo visual) |
| `xmax` | número | (-∞, ∞) | - | Etiqueta X máxima (solo visual) |
| `ancho` | int | 20-100 | 30-80 | Ancho en columnas |
| `alto` | int | 5-50 | 10-30 | Altura en filas |

**Retorna:** 0

**Uso:** Gráfica de función matemática

---

# Tabla Resumen Rápida

## Funciones Activación
| Función | Entrada | Salida | Rango Salida |
|---------|---------|--------|--------------|
| sig(x) | número | número | [0, 1] |
| relu(x) | número | número | [0, ∞) |
| tanh_dl(x) | número | número | [-1, 1] |

## Funciones Normalización
| Función | Entrada | Salida | Rango Salida |
|---------|---------|--------|--------------|
| media(datos, n) | arreglo, int | número | (-∞, ∞) |
| desviacion(datos, n) | arreglo, int | número | [0, ∞) |
| normalizar_zscore(datos, n) | arreglo, int | arreglo | (-∞, ∞) |
| normalizar_minmax(datos, n) | arreglo, int | arreglo | [0, 1] |

## Funciones Métricas
| Función | Entrada | Salida | Rango Salida |
|---------|---------|--------|--------------|
| mse(y_real, y_pred, n) | 3 arreglos | número | [0, ∞) |
| mae(y_real, y_pred, n) | 3 arreglos | número | [0, ∞) |
| r_cuadrado(y_real, y_pred, n) | 3 arreglos | número | (-∞, 1] |
| accuracy(y_real, y_pred, n) | 3 arreglos | número | [0, 1] |

## Funciones Regresión Lineal
| Función | Parámetros | Retorna |
|---------|-----------|---------|
| regresion_lineal(X, Y, n, lr, epochs) | X, Y, n, lr, epochs | [w, b] |
| rl_predecir(params, X, n) | params, X, n | arreglo |

## Funciones Regresión Logística
| Función | Parámetros | Retorna |
|---------|-----------|---------|
| regresion_logistica(X, Y, n, lr, epochs) | X, Y, n, lr, epochs | [w, b] |
| log_predecir(params, X, n) | params, X, n | arreglo |

## Funciones Red Neuronal
| Función | Parámetros | Retorna |
|---------|-----------|---------|
| ml_semilla(s) | s | - |
| mlp_init(n_in, n_hidden, n_out) | 3 int | red |
| mlp_entrenar(red, X, Y, n, lr, epochs) | red, X, Y, n, lr, epochs | red |
| mlp_predecir(red, X, n) | red, X, n | arreglo |

## Funciones K-Means
| Función | Parámetros | Retorna |
|---------|-----------|---------|
| kmeans(X, n, m, k, max_iter) | X, n, m, k, max_iter | arreglo |
| kmeans_resumen(etiquetas, n, k) | etiquetas, n, k | - |

## Funciones Gráficas
| Función | Parámetros Principales | Retorna |
|---------|-----------|---------|
| grafica_barras(datos, n, ancho) | datos, n, ancho | 0 |
| grafica_barras_v(datos, n, alto) | datos, n, alto | 0 |
| grafica_linea(datos, n, ancho, alto) | datos, n, ancho, alto | 0 |
| grafica_dispersion(X, Y, n, w, h) | X, Y, n, w, h | 0 |
| histograma(datos, n, bins, ancho) | datos, n, bins, ancho | 0 |
| grafica_pastel(datos, n) | datos, n | 0 |
| grafica_funcion(Y, n, xmin, xmax, w, h) | Y, n, xmin, xmax, w, h | 0 |


---

Generado: Mayo 2026
DeepLang v1.0 - Custom Interpreted Language with Machine Learning
