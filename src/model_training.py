"""
model_training.py
Módulo para entrenamiento de modelos de clasificación.
Contiene funciones para crear y entrenar Random Forest y Gradient Boosting.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold


def dividir_datos(X, y, test_size=0.2, random_state=42):
    """
    Divide los datos en conjunto de entrenamiento y prueba.
    Usa stratified split para preservar la distribución de clases.
    
    Parámetros:
    -----------
    X : pd.DataFrame o np.ndarray
        Features
    y : pd.Series o np.ndarray
        Target
    test_size : float
        Proporción del conjunto de prueba
    random_state : int
        Semilla aleatoria para reproducibilidad
    
    Retorna:
    --------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    
    print(f'Split realizado:')
    print(f'  Entrenamiento: {X_train.shape[0]} muestras')
    print(f'  Prueba: {X_test.shape[0]} muestras')
    
    return X_train, X_test, y_train, y_test


def entrenar_random_forest(X_train, y_train, n_estimators=100, max_depth=10, 
                           min_samples_split=5, min_samples_leaf=2, random_state=42):
    """
    Entrena un modelo de Random Forest.
    
    Parámetros:
    -----------
    X_train : pd.DataFrame o np.ndarray
        Features de entrenamiento
    y_train : pd.Series o np.ndarray
        Target de entrenamiento
    n_estimators : int
        Número de árboles
    max_depth : int
        Profundidad máxima de árboles
    min_samples_split : int
        Mínimo de muestras para dividir un nodo
    min_samples_leaf : int
        Mínimo de muestras en hoja final
    random_state : int
        Semilla aleatoria
    
    Retorna:
    --------
    RandomForestClassifier
        Modelo entrenado
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    print('Random Forest entrenado.')
    return model


def entrenar_gradient_boosting(X_train, y_train, n_estimators=100, max_depth=5, 
                               learning_rate=0.1, random_state=42):
    """
    Entrena un modelo de Gradient Boosting.
    
    Parámetros:
    -----------
    X_train : pd.DataFrame o np.ndarray
        Features de entrenamiento
    y_train : pd.Series o np.ndarray
        Target de entrenamiento
    n_estimators : int
        Número de árboles secuenciales
    max_depth : int
        Profundidad máxima de árboles
    learning_rate : float
        Tasa de aprendizaje (shrinkage)
    random_state : int
        Semilla aleatoria
    
    Retorna:
    --------
    GradientBoostingClassifier
        Modelo entrenado
    """
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state
    )
    
    model.fit(X_train, y_train)
    
    print('Gradient Boosting entrenado.')
    return model


def validacion_cruzada(model, X_train, y_train, cv=5, scoring='f1_weighted'):
    """
    Realiza validación cruzada k-fold.
    
    Parámetros:
    -----------
    model : estimator
        Modelo de sklearn
    X_train : pd.DataFrame o np.ndarray
        Features de entrenamiento
    y_train : pd.Series o np.ndarray
        Target de entrenamiento
    cv : int
        Número de pliegues
    scoring : str
        Métrica de evaluación
    
    Retorna:
    --------
    np.ndarray
        Scores de cada pliegue
    """
    cv_fold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    scores = cross_val_score(
        model, X_train, y_train, 
        cv=cv_fold, 
        scoring=scoring,
        n_jobs=-1
    )
    
    print(f'\nValidación Cruzada ({cv}-fold, {scoring}):')
    print(f'  Media: {scores.mean():.4f}')
    print(f'  Std: {scores.std():.4f}')
    print(f'  Rango: [{scores.min():.4f}, {scores.max():.4f}]')
    
    return scores


def obtener_importancia_features(model, feature_names):
    """
    Obtiene la importancia de features de un modelo de ensemble.
    
    Parámetros:
    -----------
    model : RandomForestClassifier o GradientBoostingClassifier
        Modelo entrenado
    feature_names : list
        Nombres de las features
    
    Retorna:
    --------
    pd.DataFrame
        DataFrame con importancias ordenadas
    """
    import pandas as pd
    
    importances = model.feature_importances_
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    return importance_df


if __name__ == '__main__':
    # Ejemplo de uso
    print('Módulo model_training.py')
    print('Use las funciones en sus notebooks para entrenar modelos.')
