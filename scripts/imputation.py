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


def convertir_salaires_en_euro(df, col_salaire='sd_20month', col_pays='COUNTRY'):
    """
    Convertit la colonne des salaires (monnaie locale) en Euros (base 2020)
    en utilisant un dictionnaire de taux de change fixe.
    La colonne originale est écrasée par les valeurs converties.
    """
    
    # 1. Définition des taux (1 Unité Locale = X Euros)
    conversion_rates = {
        # Zone Euro (Taux = 1)
        'Austria': 1.0, 'Belgium': 1.0, 'Cyprus': 1.0, 'Finland': 1.0, 'France': 1.0,
        'Germany': 1.0, 'Greece': 1.0, 'Ireland': 1.0, 'Italy': 1.0, 'Luxembourg': 1.0,
        'Malta': 1.0, 'Netherlands': 1.0, 'Portugal': 1.0, 'Slovakia': 1.0, 'Slovenia': 1.0,
        'Spain': 1.0, 'Catalunya': 1.0, 'Estonia': 1.0, 'Latvia': 1.0, 'Lithuania': 1.0,

        # Hors Zone Euro
        'BosniaHerzegovina': 1 / 1.956,  # BAM
        'Bulgaria': 1 / 1.956,           # BGN
        'Croatia': 1 / 7.430,            # HRK
        'Denmark': 1 / 7.472,            # DKK
        'Hungary': 1 / 330.50,           # HUF
        'Iceland': 1 / 138.83,           # ISK
        'Moldova': 1 / 19.33,            # MDL
        'Norway': 1 / 9.855,             # NOK
        'Poland': 1 / 4.256,             # PLN
        'Romania': 1 / 4.779,            # RON
        'Serbia': 1 / 117.85,            # RSD
        'Sweden': 1 / 10.467,            # SEK
        'Czech Republic': 1 / 25.40,     # CZK
        'United Kingdom': 1 / 0.844,     # GBP
        'Switzerland': 1 / 1.07,         # CHF (Ajout fréquent, à vérifier si présent)
        'North Macedonia': 1 / 61.50,    # MKD
        'Albania': 1 / 122.00,           # ALL
        'Montenegro': 1.0,               # Utilise l'Euro
        'Kosovo': 1.0,                   # Utilise l'Euro
        'Ukraine': 1 / 26.50             # UAH (Approx 2020)
    }

    # 2. Création du vecteur de taux correspondant aux pays du DataFrame
    # On utilise map() pour associer chaque ligne à son taux
    taux_applicables = df[col_pays].map(conversion_rates)

    # 4. Conversion et Remplacement
    # On écrase l'ancienne colonne par la nouvelle valeur
    df['sd_20month_EUR_2020'] = df[col_salaire] * taux_applicables
    df.drop(columns=[col_salaire], inplace=True)

    print(f"Conversion terminée. La colonne '{col_salaire}' est maintenant exprimée en EUROS.")

    return df

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