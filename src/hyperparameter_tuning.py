"""
hyperparameter_tuning.py
Módulo para optimización de hiperparámetros.
Contiene funciones para GridSearchCV y análisis de resultados.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def parametros_random_forest():
    """
    Define el espacio de búsqueda de hiperparámetros para Random Forest.
    
    Retorna:
    --------
    dict
        Grid de parámetros
    """
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [5, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    return param_grid


def parametros_gradient_boosting():
    """
    Define el espacio de búsqueda de hiperparámetros para Gradient Boosting.
    
    Retorna:
    --------
    dict
        Grid de parámetros
    """
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'min_samples_split': [2, 5, 10]
    }
    
    return param_grid


def ejecutar_grid_search_rf(X_train, y_train, param_grid=None, cv=5, scoring='f1_weighted'):
    """
    Ejecuta GridSearchCV para Random Forest.
    
    Parámetros:
    -----------
    X_train : pd.DataFrame o np.ndarray
        Features de entrenamiento
    y_train : pd.Series o np.ndarray
        Target de entrenamiento
    param_grid : dict, opcional
        Grid de parámetros. Si None, usa parametros_random_forest()
    cv : int
        Número de pliegues
    scoring : str
        Métrica de evaluación
    
    Retorna:
    --------
    GridSearchCV
        Objeto con resultados de la búsqueda
    """
    if param_grid is None:
        param_grid = parametros_random_forest()
    
    cv_fold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid,
        cv=cv_fold,
        scoring=scoring,
        n_jobs=-1,
        verbose=1
    )
    
    print(f'GridSearch RF: {np.prod([len(v) for v in param_grid.values()])} combinaciones')
    grid_search.fit(X_train, y_train)
    
    print(f'Mejores parámetros: {grid_search.best_params_}')
    print(f'Mejor score: {grid_search.best_score_:.4f}')
    
    return grid_search


def ejecutar_grid_search_gb(X_train, y_train, param_grid=None, cv=5, scoring='f1_weighted'):
    """
    Ejecuta GridSearchCV para Gradient Boosting.
    
    Parámetros:
    -----------
    X_train : pd.DataFrame o np.ndarray
        Features de entrenamiento
    y_train : pd.Series o np.ndarray
        Target de entrenamiento
    param_grid : dict, opcional
        Grid de parámetros. Si None, usa parametros_gradient_boosting()
    cv : int
        Número de pliegues
    scoring : str
        Métrica de evaluación
    
    Retorna:
    --------
    GridSearchCV
        Objeto con resultados de la búsqueda
    """
    if param_grid is None:
        param_grid = parametros_gradient_boosting()
    
    cv_fold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        GradientBoostingClassifier(random_state=42),
        param_grid,
        cv=cv_fold,
        scoring=scoring,
        n_jobs=-1,
        verbose=1
    )
    
    print(f'GridSearch GB: {np.prod([len(v) for v in param_grid.values()])} combinaciones')
    grid_search.fit(X_train, y_train)
    
    print(f'Mejores parámetros: {grid_search.best_params_}')
    print(f'Mejor score: {grid_search.best_score_:.4f}')
    
    return grid_search


def analizar_resultados_grid_search(grid_search, top_n=10):
    """
    Analiza los resultados de una búsqueda de grid.
    
    Parámetros:
    -----------
    grid_search : GridSearchCV
        Objeto de GridSearchCV completado
    top_n : int
        Cantidad de mejores configuraciones a mostrar
    
    Retorna:
    --------
    pd.DataFrame
        Top configuraciones
    """
    resultados = pd.DataFrame(grid_search.cv_results_)
    
    # Seleccionar columnas relevantes
    columnas = ['param_n_estimators', 'param_max_depth', 'mean_test_score', 'std_test_score', 'rank_test_score']
    
    # Filtrar columnas que existan
    columnas_existentes = [col for col in columnas if col in resultados.columns]
    
    top_resultados = resultados[columnas_existentes].nsmallest(top_n, 'rank_test_score')
    
    print(f'\nTop {top_n} configuraciones:')
    print(top_resultados.to_string())
    
    return top_resultados


def extraer_modelo_optimizado(grid_search):
    """
    Extrae el modelo optimizado del GridSearchCV.
    
    Parámetros:
    -----------
    grid_search : GridSearchCV
        Objeto de GridSearchCV completado
    
    Retorna:
    --------
    estimator
        Mejor modelo encontrado
    """
    mejor_modelo = grid_search.best_estimator_
    print(f'Modelo optimizado extraído.')
    
    return mejor_modelo


def comparar_base_vs_optimizado(modelo_base, modelo_opt, X_test, y_test, 
                                y_pred_base, y_pred_opt, scoring_fn):
    """
    Compara rendimiento de modelo base vs optimizado.
    
    Parámetros:
    -----------
    modelo_base : estimator
        Modelo sin optimizar
    modelo_opt : estimator
        Modelo optimizado
    X_test : pd.DataFrame o np.ndarray
        Features de prueba
    y_test : pd.Series o np.ndarray
        Target de prueba
    y_pred_base : array-like
        Predicciones del modelo base
    y_pred_opt : array-like
        Predicciones del modelo optimizado
    scoring_fn : callable
        Función de métrica (ej: accuracy_score)
    
    Retorna:
    --------
    pd.DataFrame
        Comparación de rendimiento
    """
    from sklearn.metrics import accuracy_score
    
    comparacion = pd.DataFrame({
        'Modelo': ['Base', 'Optimizado'],
        'Accuracy': [
            accuracy_score(y_test, y_pred_base),
            accuracy_score(y_test, y_pred_opt)
        ]
    })
    
    mejora = comparacion.loc[1, 'Accuracy'] - comparacion.loc[0, 'Accuracy']
    comparacion['Mejora'] = [0, mejora]
    
    print('\nComparación Base vs Optimizado:')
    print(comparacion.to_string(index=False))
    
    if mejora > 0:
        print(f'\nMejora obtenida: +{mejora:.4f} ({mejora*100:.2f}%)')
    elif mejora < 0:
        print(f'\nNo se obtuvo mejora: {mejora:.4f}')
    else:
        print('\nRendimiento similar.')
    
    return comparacion


if __name__ == '__main__':
    print('Módulo hyperparameter_tuning.py')
    print('Use las funciones en sus notebooks para optimizar modelos.')
