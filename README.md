# Proyecto 1 - Predicción de deserción y éxito académico

**Curso:** Técnicas de Aprendizaje de Máquina - Pontificia Universidad Javeriana (Bogotá)
**Profesor:** Ing. Julio Omar Palacio Niño, M.Sc.

**Integrantes**
- Diego Céspedes
- Carlos Méndez
- Simon Fajardo
- Diego Sarmiento

---

## 1. Descripción

Pipeline de Machine Learning supervisado para detectar la **deserción de estudiantes universitarios** a partir de información conocida al momento de la matrícula y del rendimiento académico durante el primer y segundo semestre. El proyecto se enfoca en el análisis exploratorio, la selección argumentada de variables y la sintonización de hiperparámetros con validación cruzada para controlar el sobreajuste.

## 2. Dataset

[Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) (UCI ML Repository, id 697).

| Característica | Valor |
|---|---|
| Estudiantes | 4.424 |
| Variables predictoras | 36 (sin valores nulos ni filas duplicadas) |
| Variable objetivo original | `Target`: Graduate (49,9 %), Dropout (32,1 %), Enrolled (17,9 %) |

El dataset se descarga automáticamente con la librería `ucimlrepo` (no hay que bajar archivos manualmente).

**Variable objetivo binarizada:** se agrupan `Enrolled` y `Graduate` en una sola clase para convertir el problema en la detección de deserción.

| Clase | Significado | Proporción |
|---|---|---|
| 1 | Dropout | ≈ 32 % |
| 0 | No deserta (Enrolled + Graduate) | ≈ 68 % |

## 3. Restricciones técnicas del enunciado

No se usan redes neuronales, ensambles (Random Forest, XGBoost, etc.), Naive Bayes ni PCA. El ajuste de hiperparámetros se hace únicamente con `GridSearchCV` + validación cruzada K-Fold, sin ensayo y error manual.

## 4. Estructura del repositorio

```
.
├── Proyecto1.ipynb     # Notebook con todo el desarrollo (EDA, selección, modelos, curvas)
└── README.md
```

## 5. Metodología

El notebook sigue el orden de las tareas del enunciado.

### 5.1 Análisis exploratorio y de dimensionalidad
- Tipificación de variables: binarias, nominales, ordinales y numéricas (los códigos de la UCI no son todos numéricos).
- Calidad de datos: nulos, duplicados y variables casi constantes (`Educational special needs`, `Nacionality`, `International`).
- Distribución del target (original y binarizado) y razón de desbalance (≈ 2,1 : 1).
- Distribuciones, asimetría, proporción de ceros y outliers (regla del IQR) de las variables numéricas.
- Matriz de correlación de variables numéricas para detectar multicolinealidad.

### 5.2 Selección de variables
1. **Filtro por correlación:** se detectan pares con |r| > 0,8 entre variables del 1.er y 2.º semestre (≈ 0,84 a 0,94). Se conservan las del 2.º semestre y se eliminan las del 1.º (`credited`, `enrolled`, `approved`, `grade`).
2. **Variables casi constantes:** se eliminan `Nacionality` y `Educational special needs`.
3. **Codificación one-hot** de las variables nominales (`Marital Status`, `Application mode`, `Course`, `Previous qualification`, `Mother's qualification`, `Father's qualification`, `Mother's occupation`).
4. **Split estratificado 80/20 antes de cualquier ajuste** (3.539 en entrenamiento / 885 en prueba), para evitar fuga de datos.
5. **Regularización L1 (Lasso)** ajustada solo con el conjunto de entrenamiento: se descartan las variables con coeficiente 0, y el conjunto final queda en **103 variables**.

### 5.3 Modelado dual

| Rol | Algoritmo | Motivo |
|---|---|---|
| Baseline | Regresión Logística | Máxima interpretabilidad (coeficientes) |
| Retador | SVM (kernels RBF y polinomial) | Frontera no lineal para intentar superar al baseline |

Ambos modelos usan un `Pipeline` (`StandardScaler` + clasificador) para que el escalado se ajuste dentro de cada fold, `class_weight='balanced'`, `StratifiedKFold` de 5 particiones y `scoring='f1_weighted'`.

**Espacio de búsqueda**
- Regresión Logística: penalización L1 / L2 / ElasticNet, `C` ∈ [0,001 ... 100], solvers `lbfgs` y `saga`, `l1_ratio` ∈ {0,2; 0,5; 0,7}.
- SVM: RBF con `C` ∈ {0,1; 1; 10; 100} y `gamma` ∈ {scale; 0,001; 0,01; 0,1}; polinomial con `degree` ∈ {2, 3}, `C`, `gamma` y `coef0`.

### 5.4 Curvas de validación y de aprendizaje (punto 4b)
- **Curvas de validación** (train vs. CV, con banda de desviación entre folds) para `C` de la regresión logística y para `C` y `gamma` de la SVM-RBF, con los demás hiperparámetros fijos en su mejor valor.
- **Curvas de aprendizaje** de ambos modelos.
- Se interpretan en términos de **sesgo y varianza**: `C` alto o `gamma` alto aumentan la varianza (el gap entre train y CV crece), mientras que `C` bajo o `gamma` bajo aumentan el sesgo.

## 6. Resultados

Evaluación sobre el conjunto de prueba (885 estudiantes, 284 de ellos desertores).

| Modelo | Mejores hiperparámetros | F1-weighted (CV) | Accuracy (test) | F1 macro (test) | F1 Dropout (test) |
|---|---|---|---|---|---|
| Regresión Logística (baseline) | `C=0.01`, `penalty=l2`, `solver=lbfgs` | 0,867 | 0,88 | 0,86 | 0,81 |
| SVM RBF (retador) | `C=10`, `gamma=0.001` | 0,870 | 0,88 | 0,86 | 0,81 |

**Principales hallazgos**
- Las variables más influyentes para el Lasso son del 2.º semestre (unidades aprobadas e inscritas) y `Tuition fees up to date`.
- Ambos modelos llegan al mismo desempeño (≈ 0,87 en CV); el retador no supera de forma significativa al baseline.
- Las curvas de aprendizaje muestran gaps pequeños (≈ 0,01 en la regresión logística y ≈ 0,02 en la SVM) y convergencia en un valor similar: los modelos están limitados más por el **sesgo y la información de las variables** que por la varianza, por lo que conseguir más datos aportaría poco.
- En la SVM, un `gamma` alto genera sobreajuste severo (train ≈ 1,0 y CV por debajo de 0,6).

## 7. Cómo ejecutar

**Requisitos**
- Python 3.10 o superior
- Jupyter Notebook, JupyterLab o VS Code con soporte para notebooks

**Instalación**

```bash
python -m venv entorno_proyecto
# Windows
entorno_proyecto\Scripts\activate
# Linux / macOS
source entorno_proyecto/bin/activate

pip install ucimlrepo pandas numpy matplotlib seaborn scipy scikit-learn jupyter
```

**Ejecución**

1. Abrir `Proyecto1.ipynb`.
2. Ejecutar **Kernel > Restart & Run All** (se necesita conexión a internet para descargar el dataset de la UCI).
3. Las celdas de `GridSearchCV` pueden tardar varios minutos (usan `n_jobs=-1`).

Todas las semillas aleatorias están fijadas (`random_state=42`), por lo que los resultados son reproducibles.

## 8. Limitaciones

- Al agrupar `Enrolled` con `Graduate`, el modelo no distingue entre estudiantes que siguen matriculados y quienes ya se graduaron.
- El desempeño depende de variables del 2.º semestre, por lo que la detección ocurre una vez avanzado el año académico y no al momento de la matrícula.
- El techo de rendimiento (≈ 0,87 en CV) sugiere que hace falta ingeniería de características adicional (por ejemplo, tasas de aprobación o tendencias entre semestres) para mejorar.

## 9. Referencia

Realinho, V., Vieira Martins, M., Machado, J., Baptista, L. (2021). *Predict Students' Dropout and Academic Success*. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89
