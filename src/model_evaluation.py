"""
model_evaluation.py
Módulo para evaluación de modelos de clasificación.
Contiene funciones para calcular métricas, matrices de confusión y reportes.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)


def calcular_metricas(y_true, y_pred, promedio='weighted'):
    """
    Calcula múltiples métricas de clasificación.
    
    Parámetros:
    -----------
    y_true : array-like
        Etiquetas verdaderas
    y_pred : array-like
        Etiquetas predichas
    promedio : str
        Tipo de promedio para multiclase ('weighted', 'macro', etc.)
    
    Retorna:
    --------
    dict
        Diccionario con métricas
    """
    metricas = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=promedio, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=promedio, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=promedio, zero_division=0)
    }
    
    return metricas


def obtener_confusion_matrix(y_true, y_pred):
    """
    Obtiene la matriz de confusión.
    
    Parámetros:
    -----------
    y_true : array-like
        Etiquetas verdaderas
    y_pred : array-like
        Etiquetas predichas
    
    Retorna:
    --------
    np.ndarray
        Matriz de confusión
    """
    cm = confusion_matrix(y_true, y_pred)
    return cm


def obtener_reporte_clasificacion(y_true, y_pred, target_names=None):
    """
    Obtiene el reporte completo de clasificación por clase.
    
    Parámetros:
    -----------
    y_true : array-like
        Etiquetas verdaderas
    y_pred : array-like
        Etiquetas predichas
    target_names : list, opcional
        Nombres de las clases
    
    Retorna:
    --------
    str
        Reporte formateado
    """
    report = classification_report(
        y_true, y_pred, 
        target_names=target_names,
        zero_division=0
    )
    
    return report


def comparar_modelos(modelos_dict, X_test, y_test):
    """
    Compara múltiples modelos en el conjunto de prueba.
    
    Parámetros:
    -----------
    modelos_dict : dict
        Diccionario {nombre: modelo}
    X_test : pd.DataFrame o np.ndarray
        Features de prueba
    y_test : pd.Series o np.ndarray
        Target de prueba
    
    Retorna:
    --------
    pd.DataFrame
        DataFrame comparativo con métricas
    """
    resultados = []
    
    for nombre, modelo in modelos_dict.items():
        y_pred = modelo.predict(X_test)
        metricas = calcular_metricas(y_test, y_pred)
        
        resultados.append({
            'Modelo': nombre,
            'Accuracy': metricas['accuracy'],
            'Precision': metricas['precision'],
            'Recall': metricas['recall'],
            'F1': metricas['f1']
        })
    
    df_comparacion = pd.DataFrame(resultados)
    
    print('\nComparación de Modelos:')
    print(df_comparacion.to_string(index=False))
    
    return df_comparacion


def diagnosticar_overfitting(train_score, test_score, threshold=0.15):
    """
    Diagnostica si hay overfitting comparando train vs test scores.
    
    Parámetros:
    -----------
    train_score : float
        Score en conjunto de entrenamiento
    test_score : float
        Score en conjunto de prueba
    threshold : float
        Brecha máxima aceptable
    
    Retorna:
    --------
    str
        Diagnóstico
    """
    brecha = train_score - test_score
    
    if brecha > threshold:
        diagnostico = f'OVERFITTING DETECTADO (brecha={brecha:.4f})'
    elif brecha < -0.05:
        diagnostico = 'UNDERFITTING (test mejor que train - poco probable)'
    else:
        diagnostico = 'MODELO BIEN AJUSTADO'
    
    return diagnostico


def analizar_errores_por_clase(y_true, y_pred, classes):
    """
    Analiza errores desglosados por clase.
    
    Parámetros:
    -----------
    y_true : array-like
        Etiquetas verdaderas
    y_pred : array-like
        Etiquetas predichas
    classes : list
        Nombres de las clases
    
    Retorna:
    --------
    pd.DataFrame
        Análisis de errores por clase
    """
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    errores = []
    for i, clase in enumerate(classes):
        total = cm[i].sum()
        correctos = cm[i, i]
        incorrectos = total - correctos
        tasa_error = incorrectos / total if total > 0 else 0
        
        errores.append({
            'Clase': clase,
            'Total': total,
            'Correctos': correctos,
            'Incorrectos': incorrectos,
            'Tasa Error': f'{tasa_error:.2%}'
        })
    
    df_errores = pd.DataFrame(errores)
    
    return df_errores


def obtener_predicciones_confianza(modelo, X):
    """
    Obtiene predicciones con probabilidades (confianza).
    
    Parámetros:
    -----------
    modelo : estimator
        Modelo con método predict_proba
    X : pd.DataFrame o np.ndarray
        Features
    
    Retorna:
    --------
    tuple
        (y_pred, probabilidades)
    """
    y_pred = modelo.predict(X)
    probabilidades = modelo.predict_proba(X)
    confianza_maxima = probabilidades.max(axis=1)
    
    return y_pred, confianza_maxima


if __name__ == '__main__':
    print('Módulo model_evaluation.py')
    print('Use las funciones en sus notebooks para evaluar modelos.')
