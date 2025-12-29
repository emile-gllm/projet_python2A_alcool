import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
import statsmodels.api as sm
from sklearn.ensemble import StackingRegressor, RandomForestRegressor


def imputer_salaire_pays(df, colonne_salaire='sd_20month_EUR_2020', colonne_pays='COUNTRY', liste_pays = ['France', 'Iceland', 'Bulgaria', 'Poland']):
    """
    Parcourt tous les pays de la base. Pour chaque pays, entraîne un modèle 
    spécifique sur les données de ce pays pour imputer ses valeurs manquantes.
    """
    
    
    print(f"Début de l'imputation pour {len(liste_pays)} pays...\n")

    # On boucle sur chaque pays
    for pays_cible in liste_pays:
        
        # Filtrage des données pour ce pays
        # On utilise les indices pour pouvoir modifier le df original plus tard
        indices_pays = df[df[colonne_pays] == pays_cible].index
        df_pays = df.loc[indices_pays]
        
        # Séparer données connues (Train) et manquantes (Predict)
        train_data = df_pays[df_pays[colonne_salaire].notna()]
        predict_data = df_pays[df_pays[colonne_salaire].isna()]
        
        # Si aucune valeur manquante pour ce pays, on passe au suivant
        if predict_data.empty:
            continue
            
        # Préparation des variables (X et y)
        # On exclut la cible et la colonne pays (car elle est constante ici)
        features = df_pays.drop(columns=[colonne_salaire, colonne_pays])
        
       # Préparation des X et y pour l'entraînement
        X = features.loc[train_data.index]
        y = np.log(train_data[colonne_salaire])
        
        # 3. Évaluation des performances (Validation croisée simple)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=200, 
        max_depth=7,              
        min_samples_leaf=5,       
        max_features='sqrt',      
        random_state=42)

        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)
        
        print(f"--- Performances du modèle pour {pays_cible} ---")
        print(f"Score R² : {r2_score(y_test, y_pred_test):.4f}")
        print(f"RMSE : {np.sqrt(mean_squared_error(y_test, y_pred_test)):.2f}")
        print("--------------------------------------------\n")
        
        
        # 4. Imputation réelle
        # Ré-entraînement sur toute la donnée disponible du pays
        model.fit(X, y)
        X_miss = features.loc[predict_data.index]
        predictions = model.predict(X_miss)

        
        # Insertion des valeurs prédites dans le DataFrame original
        df.loc[predict_data.index, colonne_salaire] = np.exp(predictions)

    
    print("\n--- Imputation terminée pour tous les pays ---")
    return df



def imputer_knn(df, cols_cibles=['RSOD_2b', 'SD_7'], col_pays='COUNTRY', n_voisins=5, liste_pays=['France', 'Iceland', 'Bulgaria', 'Poland']):
    """
    Parcourt les pays et impute toutes les colonnes listées dans 'cols_cibles' 
    en utilisant KNN.
    """    

    for pays_cible in liste_pays:
        # 1. Isolation du pays
        indices_pays = df[df[col_pays] == pays_cible].index
        # On travaille sur une copie
        df_pays = df.loc[indices_pays].copy()
        
        # Vérification rapide : Si TOUTES les colonnes cibles sont pleines, on passe
        nb_na_total = df_pays[cols_cibles].isna().sum().sum()
        if nb_na_total == 0:
            continue
            
        # 2. Préparation des données (Tout sauf le pays)
        # KNN a besoin de TOUTES les variables numériques pour trouver les voisins
        features_cols = [c for c in df_pays.columns if c != col_pays]
        X = df_pays[features_cols]
        

        # 3. Normalisation 
        scaler = StandardScaler()
        X_scaled_values = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled_values, columns=features_cols, index=X.index)

        # 4. Imputation Globale 
        imputer = KNNImputer(n_neighbors=n_voisins)
        X_imputed_values = imputer.fit_transform(X_scaled)
        
        # 5. Dé-normalisation
        X_final_values = scaler.inverse_transform(X_imputed_values)
        X_final = pd.DataFrame(X_final_values, columns=features_cols, index=X.index)
        
        # 6. Injection boucle par boucle 
        # On parcourt chaque colonne qu'on voulait réparer
        for col in cols_cibles:
            if col not in df_pays.columns:
                continue
                
            # On ne touche qu'aux lignes qui étaient vides pour CETTE colonne
            indices_manquants = df_pays[df_pays[col].isna()].index
            
            if len(indices_manquants) > 0:
                # On récupère les valeurs calculées et on arrondit 
                valeurs_imputees = X_final.loc[indices_manquants, col].round()
                
                # Mise à jour dans le DataFrame Principal
                df.loc[indices_manquants, col] = valeurs_imputees

    print(f"\n Imputation terminée.")
    return df