# RESULTS — resultados que necesita el Capítulo 5

Lista de los resultados que hay que extraer para tener todo lo que el capítulo de validación
necesita, el orden lógico en que deben ejecutarse, y qué falta programar. Reemplaza a
`PLOTS.md`. La estructura del capítulo está en `../report/5-validacion.tex`; el *qué valida
cada cosa* en `../report/planificacion/obtencion-de-resultados.md`; la organización de los
scripts en `README.md` ("Layout").

Todo se corre desde `backend/` con `uv run -m …`.

## Convenciones

- La salida de evaluación va a **`results/<modelo>/…`** (`RESULTS_DIR`, gitignored). Los
  `.tflite` servibles quedan en `shared/gen/models/<modelo>/` y **no** son resultados.
- Cada `<nombre>.png` lleva un **`<nombre>.yaml`** al lado (qué muestra, ejes y unidades,
  sujetos y splits, números principales, qué sección respalda), así que el resultado se lee y
  se cita sin abrir la imagen. Los numéricos llevan además `.csv`.
- **Todo bucle retiene sujetos completos** (`--eval-subjects`, por defecto `14-15`), y los ids
  resueltos quedan en el manifiesto `run.yaml`. Toda métrica del capítulo es generalización a
  un sujeto no visto.
- Los scripts se dividen en dos clases: los que **leen** una corrida previa
  (`plot_convergence`) y los que **entrenan** porque barren configuraciones que ninguna
  corrida de `train.py` produce (`byzantine`, `sensitivity`, `knowledge_distillation`). Los
  segundos son pesados: lanzarlos en segundo plano.
- Las curvas del informe salen de los **bucles simulados** (reproducibles, semilla en
  `scripts/__init__.py`), no del cliente headless sobre HTTP, que es verificación de
  integración.

---

## Orden de ejecución

Las etapas están en orden de dependencia: cada una supone hechas las anteriores.

### 0. Dataset

Idempotente; salta descarga y procesamiento si ya está.

```bash
uv run -m scripts.system.get_dataset
```

### 1. Corridas de entrenamiento base

Fuente de las curvas de la Sec. 5.2 y de los pesos que consumen casi todas las etapas
siguientes. Los dos bucles con el **mismo** `--eval-subjects`, o el solapamiento no compara
lo mismo (`plot_convergence` se niega a dibujarlo si los manifiestos discrepan).

```bash
uv run -m scripts.system.train cnn-ae      --loop normal    --eval-subjects 14-15 --epochs 10
uv run -m scripts.system.train cnn-ae      --loop federated --eval-subjects 14-15 --epochs 10
uv run -m scripts.system.train feature-mlp --loop normal    --eval-subjects 14-15
uv run -m scripts.system.train feature-mlp --loop federated --eval-subjects 14-15
```

| Salida | Alimenta |
|--------|----------|
| `results/<modelo>/<loop>/{training.png,training.csv,run.yaml}` (+ `reconstruction.png` en los AE) | insumo de 1.1; el reporte de evaluación se cita en 5.2 |

Estado actual (`cnn-ae`, 10 épocas, lote 64, `recon_error` 0,153 sobre S14-S15): el punto de
operación calibrado es $f = 0{,}26$, con recall 0,401, precisión 0,637, exactitud 0,582 y
$J = 0{,}140$. La corrida anterior de 5 épocas daba $J = 0{,}082$ en $f = 0{,}45$, así que el
detector casi duplicó su separación y además su punto de operación pasó a ser razonable. Las
cifras del capítulo pueden escribirse contra esta corrida.

#### 1.1 Figuras de convergencia (no entrenan)

```bash
uv run -m scripts.figures.plot_convergence cnn-ae
uv run -m scripts.figures.plot_convergence feature-mlp
```

| Salida | Sección |
|--------|---------|
| `results/<modelo>/federated/convergence.{png,csv,yaml}` | **5.2** — el modelo federado mejora ronda a ronda |
| `results/<modelo>/centralized_vs_federated/centralized_vs_federated.{png,csv,yaml}` | **5.2** — FedAvg alcanza calidad comparable sin centralizar (claim central) |

En el informe la Sec. 5.2 usa **una** figura, el solapamiento; la curva federada sola es una
de sus dos series. La otra va al anexo si aporta.

### 2. Barridos que entrenan

Independientes entre sí y de la etapa 3; los dos son pesados.

```bash
uv run -m scripts.figures.byzantine   cnn-ae --max-malicious 4 --rounds 5 --eval-subjects 2
uv run -m scripts.figures.sensitivity cnn-ae --sweep all --rounds 5 --eval-subjects 2
```

| Salida | Sección |
|--------|---------|
| `results/<modelo>/byzantine/byzantine.{png,csv,yaml}` | **5.3** — la media recortada sostiene la ronda donde el promedio liso no |
| `results/<modelo>/sensitivity/loso.{png,csv,yaml}` | **5.4** — figura del cuerpo: métrica final por fold, media ± desviación |
| `results/<modelo>/sensitivity/{participants,local_epochs}.{png,csv,yaml}` | **5.4** — una frase en el cuerpo, figuras al anexo |

`byzantine` corre su propio bucle federado (tiene que anexar los deltas maliciosos antes de
agregar) y barre cada agregador con el filtro z-score activado y desactivado, que son las dos
líneas de la figura. El promedio ponderado no es una opción: bajo este modelo de amenaza es
inaplicable, no meramente débil.

### 3. Huella de sistema

```bash
uv run -m scripts.figures.footprint
```

| Salida | Sección |
|--------|---------|
| `results/footprint/footprint.{csv,yaml}` | **5.5** — conteo de parámetros, float32 vs. int8 y ratio, bytes por ronda |

Las filas restantes de la tabla son mediciones puntuales que se pegan a mano (ver "Falta
programar", ítems 1 y 2).

### 4. Caso de uso: detección (maestro con split)

Consume los pesos de `cnn-ae` de la etapa 1 (el `trainable.tflite` canónico, entrenado con
split). Ninguno de los tres entrena.

```bash
uv run -m scripts.figures.calibrate_fpr     cnn-ae
uv run -m scripts.figures.anomaly_detection cnn-ae
uv run -m scripts.figures.plot_signals      cnn-ae
```

| Salida | Sección |
|--------|---------|
| `results/<modelo>/calibrate_fpr/calibration.{png,csv,yaml}` | **Cap. 4** (Fig. de calibración) — recall / FPR empírico / índice J vs. FPR esperado, con el punto elegido marcado |
| `results/<modelo>/calibrate_fpr/roc.{png,yaml}` | anexo — curva ROC del detector sobre los sujetos retenidos |
| `results/<modelo>/anomaly_detection.yaml` | **5.6.1** — precisión/recall/F1, recall por tipo de anomalía, FPR empírico en limpio |
| `results/<modelo>/{signals,signals_reconstructed}.{png,yaml}` | **Cap. 4** + Anexo de anomalías sintéticas |

### 5. Caso de uso: destilación (maestro sobre todos los usuarios)

`knowledge_distillation` y `subject_roc` quieren un maestro entrenado sobre **todos** los
sujetos, para que la calidad de las etiquetas sea uniforme y ningún fold sea especial.
Entrenarlo sobrescribe el `trainable.tflite` canónico, así que hay que apartarlo y después
restaurar el maestro con split.

```bash
uv run -m scripts.system.train cnn-ae --eval-subjects none --epochs 10 # crea artefacto llamado trainable_all.tflite

uv run -m scripts.figures.subject_roc cnn-ae \
    --weights shared/gen/models/cnn-ae/trainable_all.tflite
uv run -m scripts.figures.knowledge_distillation cnn-ae --student feature-mlp \
    --weights shared/gen/models/cnn-ae/trainable_all.tflite
```

| Salida | Sección |
|--------|---------|
| `results/<modelo>/subject_roc/{roc_by_subject,roc_aggregate}.{png,yaml}` | **5.6.1** — figura del cuerpo: dispersión de detectabilidad entre sujetos, media ± desviación |
| `results/feature-mlp/personalization/personalization.{csv,yaml}` | **5.6.2** — LOSO: global vs. personalizado, float vs. int8, más la cota superior `direct_float` (estudiante entrenado con etiquetas verdaderas) |

Estado actual (maestro `cnn-ae` sobre todos los sujetos, $f = 0{,}22$, lote 128), F1 agregado:

| variante | F1 | delta contra global |
|----------|----|---------------------|
| `global_float` | 0,347 | — |
| `global_int8` | 0,343 | |
| `personal_float` | 0,387 | $+0{,}040$ |
| `personal_int8` | 0,385 | $+0{,}042$ |
| `direct_float` | **0,664** | $+0{,}317$ |

El costo de destilación domina todo lo demás: $+0{,}317$ agregado, $+0{,}342$ de media por
fold (desv. 0,164) y mejora en **15/15**. El esquema es débil porque el maestro produce malas
etiquetas, cosa esperable con su ROC en $J = 0{,}14$. La atribución se apoya en la dispersión
y no solo en la media: `direct_float` tiene desviación por fold **0,083** contra **0,189** de
`global_float`, y entre dos corridas con maestros distintos la cota directa se movió de 0,666
a 0,664 mientras el destilado caía de 0,437 a 0,347. El estudiante es un aprendiz estable;
toda la varianza entra por el maestro.

int8 se mantiene dentro de 0,004 de F1 respecto de float en las dos variantes que lo tienen.

El último `train` no es opcional: las Secs. 5.2 y 5.6.1 quieren de vuelta el maestro con
split como artefacto canónico.

### 6. Verificación de integración (no produce figuras)

Requiere el stack arriba (PostgreSQL, Redis, api, worker) y la base sembrada (`make db-seed`).
Alimenta un párrafo de la Sec. 5.1, no una figura.

```bash
uv run -m scripts.system.seed_db --test-users
uv run -m scripts.integration.fed_client --model feature-mlp --rounds 5 --eval-subjects 2  # denso
uv run -m scripts.integration.fed_client --model cnn-ae      --rounds 5 --eval-subjects 2  # seguro
uv run -m scripts.integration.secure_aggregation --clients 4 --rounds 3
```

Lo que se reporta de aquí es cualitativo: el camino desplegado corre extremo a extremo con
varios usuarios sobre la API real, y las máscaras de la agregación segura cancelan
exactamente. Las series que escriben (`results/<modelo>/fed_client/`) no son las curvas del
informe.

---

## Falta programar

Ordenado por cuánto le cuesta al capítulo que no exista.

**1. Tiempo de ronda de agregación en el servidor (Sec. 5.5).** No lo mide ningún script.
`footprint.py` lo lista como fila para pegar a mano. Lo más barato es cronometrar la tarea de
agregación desde `scripts.integration.queue_aggregation`, que ya bloquea esperando el
resumen de la ronda, y anotar el número.

**2. Filas de teléfono y ESP32 (Sec. 5.5).** Mediciones puntuales fuera del backend, que se
pegan en la tabla de huella: tiempo de entrenamiento por época en el teléfono (logcat),
tamaño de arena de TFLM y latencia de inferencia int8 en el ESP32. No hay script que las
produzca ni lo habrá; son una corrida instrumentada en cada dispositivo.

**3. Calidad retenida tras cuantizar, medida en el dispositivo (Sec. 5.5).**
`knowledge_distillation.py` ya puntúa al estudiante en float y en int8 en el host, que
cubre la afirmación "int8 ≈ float" para el informe. Si se quiere la cifra medida sobre el
ESP32 en vez de sobre el intérprete del host, es otra medición puntual del firmware.
