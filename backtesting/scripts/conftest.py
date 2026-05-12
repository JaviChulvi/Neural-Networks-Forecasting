"""Opciones CLI personalizadas para los tests de backtesting.

Permite parametrizar los tests de predicción RNN desde la línea de comandos:

    pytest test_rnn_predictions_vary.py --model-type gru --input-window 10 --output-window 90 -v
"""


def pytest_addoption(parser):
    parser.addoption(
        "--model-type",
        default="lstm",
        choices=["lstm", "gru"],
        help="Tipo de modelo RNN a testear: lstm o gru  (default: lstm)",
    )
    parser.addoption(
        "--input-window",
        type=int,
        default=10,
        help="Tamaño de la ventana de entrada en días  (default: 10)",
    )
    parser.addoption(
        "--output-window",
        type=int,
        default=90,
        help="Tamaño de la ventana de salida en días  (default: 90)",
    )
