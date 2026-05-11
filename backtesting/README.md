# Backtesting: Modelos vs Estrategias Clásicas

Comparativa de estrategias de inversión basadas en los modelos de deep learning entrenados (MLP, RNN, CNN, mixtos) frente a benchmarks clásicos.

## Estrategias a comparar

### Benchmarks clásicos
- **Buy & Hold** — mantener el activo durante todo el periodo de test
- **Momentum** — comprar cuando el precio sube durante N sesiones consecutivas, vender cuando baja
- **Reversión a la media** — operar en contra de la tendencia reciente asumiendo retorno a la media (ej. bandas de Bollinger, z-score)

### Estrategias basadas en modelos
- Señal de compra/venta generada a partir de la predicción del modelo (precio predicho > precio actual → comprar)
- Variantes con umbral de confianza o filtro de volatilidad

## Métricas de evaluación

| Métrica | Descripción |
|---|---|
| Retorno total | Ganancia/pérdida acumulada en el periodo |
| Sharpe ratio | Retorno ajustado por riesgo (vs tasa libre de riesgo) |
| Max drawdown | Caída máxima desde un pico hasta el mínimo siguiente |
| Win rate | % de operaciones ganadoras |
| Nº de operaciones | Actividad de trading de la estrategia |

## Estructura

```
backtesting/
├── README.md
├── notebooks/          # Análisis exploratorios y comparativas
└── scripts/            # Implementaciones reutilizables de cada estrategia
```
