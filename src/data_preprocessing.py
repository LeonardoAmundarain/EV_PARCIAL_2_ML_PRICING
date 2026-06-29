"""
data_preprocessing.py
Módulo de preprocesamiento de datos para el dataset de modelos LLM.
Contiene funciones para limpieza, imputación y transformación de variables.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def cargar_dataset(ruta_archivo):
    """
    Carga el dataset desde CSV.
    
    Parámetros:
    -----------
    ruta_archivo : str
        Ruta al archivo CSV
    
    Retorna:
    --------
    pd.DataFrame
        Dataset cargado
    """
    df = pd.read_csv(ruta_archivo)
    print(f'Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas')
    return df


def eliminar_outliers_extremos(df, columnas_numericas, threshold_percentil=99):
    """
    Elimina valores extremos (outliers) en percentiles especificados.
    
    Parámetros:
    -----------
    df : pd.DataFrame
        Dataset
    columnas_numericas : list
        Columnas a evaluar
    threshold_percentil : int
        Percentil superior para considerar outlier
    
    Retorna:
    --------
    pd.DataFrame
        Dataset sin outliers extremos
    """
    df_limpio = df.copy()
    
    for col in columnas_numericas:
        if col in df_limpio.columns:
            q_high = df_limpio[col].quantile(threshold_percentil / 100)
            df_limpio = df_limpio[df_limpio[col] <= q_high]
    
    return df_limpio


def imputar_nulos_por_proveedor(df, columnas_imputacion):
    """
    Imputa valores nulos usando la mediana por proveedor.
    Esto preserva la estructura técnica de cada proveedor.
    
    Parámetros:
    -----------
    df : pd.DataFrame
        Dataset
    columnas_imputacion : list
        Columnas con valores a imputar
    
    Retorna:
    --------
    pd.DataFrame
        Dataset con nulos imputados
    """
    df_imputado = df.copy()
    
    for col in columnas_imputacion:
        if col in df_imputado.columns:
            # Imputar por proveedor primero
            df_imputado[col] = df_imputado.groupby('provider')[col].transform(
                lambda x: x.fillna(x.median())
            )
            # Luego imputar por mediana global
            df_imputado[col] = df_imputado[col].fillna(df_imputado[col].median())
    
    return df_imputado


def asignar_costo_open_source(df):
    """
    Asigna costo 0 a modelos open source.
    Los modelos open source no tienen costo de API.
    
    Parámetros:
    -----------
    df : pd.DataFrame
        Dataset
    
    Retorna:
    --------
    pd.DataFrame
        Dataset con costos corregidos para open source
    """
    df_corregido = df.copy()
    
    mask_open_source = df_corregido['is_open_source'] == True
    
    if 'input_cost_usd_per_1m' in df_corregido.columns:
        df_corregido.loc[mask_open_source, 'input_cost_usd_per_1m'] = \
            df_corregido.loc[mask_open_source, 'input_cost_usd_per_1m'].fillna(0)
    
    if 'output_cost_usd_per_1m' in df_corregido.columns:
        df_corregido.loc[mask_open_source, 'output_cost_usd_per_1m'] = \
            df_corregido.loc[mask_open_source, 'output_cost_usd_per_1m'].fillna(0)
    
    return df_corregido


def crear_features_derivadas(df):
    """
    Crea nuevas features a partir de variables existentes.
    
    Parámetros:
    -----------
    df : pd.DataFrame
        Dataset
    
    Retorna:
    --------
    pd.DataFrame
        Dataset con nuevas features
    """
    df_features = df.copy()
    
    # Costo promedio
    if 'input_cost_usd_per_1m' in df_features.columns and 'output_cost_usd_per_1m' in df_features.columns:
        df_features['cost_avg'] = (
            df_features['input_cost_usd_per_1m'] + df_features['output_cost_usd_per_1m']
        ) / 2
    
    # Inteligencia por dólar
    if 'aa_intelligence_index' in df_features.columns and 'cost_avg' in df_features.columns:
        df_features['intelligence_per_dollar'] = df_features['aa_intelligence_index'] / (
            df_features['cost_avg'] + 0.001  # Evitar división por cero
        )
    
    # Velocidad por dólar
    if 'output_tokens_per_second' in df_features.columns and 'cost_avg' in df_features.columns:
        df_features['speed_per_dollar'] = df_features['output_tokens_per_second'] / (
            df_features['cost_avg'] + 0.001
        )
    
    return df_features


def seleccionar_features(df, features_list, target_column):
    """
    Selecciona features y target, eliminando filas con nulos.
    
    Parámetros:
    -----------
    df : pd.DataFrame
        Dataset
    features_list : list
        Columnas a usar como features
    target_column : str
        Columna objetivo
    
    Retorna:
    --------
    tuple
        (X, y) - Features y target sin nulos
    """
    df_modelo = df[features_list + [target_column]].dropna()
    
    X = df_modelo[features_list]
    y = df_modelo[target_column]
    
    print(f'Dataset para modelado: {X.shape[0]} muestras, {X.shape[1]} features')
    print(f'Distribución del target:')
    print(y.value_counts())
    
    return X, y


def escalar_features(X, scaler=None):
    """
    Escala features usando StandardScaler.
    
    Parámetros:
    -----------
    X : pd.DataFrame o np.ndarray
        Features a escalar
    scaler : StandardScaler, opcional
        Scaler previamente ajustado. Si None, crea uno nuevo.
    
    Retorna:
    --------
    tuple
        (X_escalado, scaler) - Features escaladas y el scaler ajustado
    """
    if scaler is None:
        scaler = StandardScaler()
        X_escalado = scaler.fit_transform(X)
    else:
        X_escalado = scaler.transform(X)
    
    X_escalado = pd.DataFrame(X_escalado, columns=X.columns, index=X.index)
    
    return X_escalado, scaler


def procesar_dataset_completo(ruta_archivo, features_list, target_column):
    """
    Pipeline completo de preprocesamiento.
    
    Parámetros:
    -----------
    ruta_archivo : str
        Ruta al CSV
    features_list : list
        Features a usar
    target_column : str
        Columna objetivo
    
    Retorna:
    --------
    tuple
        (X, y, scaler) - Features escaladas, target y scaler
    """
    # Cargar
    df = cargar_dataset(ruta_archivo)
    
    # Eliminar nulos críticos
    df = df.dropna(subset=['aa_intelligence_index', 'aa_coding_index'], how='all')
    
    # Asignar costo 0 a open source
    df = asignar_costo_open_source(df)
    
    # Columnas para imputación
    columnas_imputacion = [
        'aa_intelligence_index', 'aa_coding_index', 'aa_math_index',
        'input_cost_usd_per_1m', 'output_cost_usd_per_1m',
        'output_tokens_per_second', 'time_to_first_token_s',
        'chatbot_arena_elo', 'release_year'
    ]
    
    # Imputar por proveedor
    df = imputar_nulos_por_proveedor(df, columnas_imputacion)
    
    # Crear features derivadas
    df = crear_features_derivadas(df)
    
    # Seleccionar features y target
    X, y = seleccionar_features(df, features_list, target_column)
    
    # Escalar
    X_escalado, scaler = escalar_features(X)
    
    return X_escalado, y, scaler


if __name__ == '__main__':
    # Ejemplo de uso
    ruta = '../datos/llm_price_performance_tracker_20260331.csv'
    features = ['aa_intelligence_index', 'aa_coding_index', 'aa_math_index',
                'input_cost_usd_per_1m', 'output_cost_usd_per_1m',
                'output_tokens_per_second', 'time_to_first_token_s',
                'chatbot_arena_elo', 'release_year', 'cost_avg']
    target = 'pricing_tier'
    
    X, y, scaler = procesar_dataset_completo(ruta, features, target)
    print('Preprocesamiento completado.')
