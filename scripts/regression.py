import pandas as pd
from IPython.display import display
import statsmodels.api as sm

def regression_pays(df,feature_cols, pays_cible, target_col ='bsqf_alc', country_col='COUNTRY'):
    """
    Effectue des régressions linéaires OLS pour une base globale et des pays spécifiques.
    
    Args:
        df (pd.DataFrame): La base de données nettoyée.
        target_col (str): Le nom de la variable dépendante (ex: 'Consommation_Alcool').
        feature_cols (list): Liste des variables explicatives (ex: ['Age', 'Salaire']).
        country_col (str): Le nom de la colonne contenant les pays.
        pays_liste (str): pays à analyser 
        
    Returns:
        dictionnaire: récapitulatif avec Coef (p-value) et élément de mesure de la qualité du modèle.
    """
    
    data_sub = df[df[country_col] == pays_cible].copy()
        
    # Préparation des variables X et y
    y = data_sub[target_col]
    X = data_sub[feature_cols]
    
    # Ajout de la constante 
    X = sm.add_constant(X)
    
    # Ajustement du modèle
    model = sm.OLS(y, X).fit()
    
    # 4. Formatage des résultats : "Coef (P-value)"
    # On parcourt chaque variable explicative + la constante 
    colonne_formattee = {}
    colonne_formattee['Pays'] = pays_cible
    for variable in X.columns:
        coef = model.params[variable]
        pval = model.pvalues[variable]
        
        colonne_formattee[variable] = f"{coef:.2f} ({pval:.3f})"
    
    # Ajout du R-carré pour information 
    colonne_formattee['R-carre'] = f"{model.rsquared:.3f}"
    
    return colonne_formattee




def afficher(donnees):
    """
    Affiche un tableau propre style Excel.
    """
    
    #  Conversion en DataFrame
    if isinstance(donnees, pd.DataFrame):
        df = donnees.copy()
    else:
        if isinstance(donnees, dict):
            donnees = [donnees]
        df = pd.DataFrame(donnees)
    
    # LOGIQUE DE TRI (Force R-squared à la fin)
    col_ref = 'Variable'
    
    if col_ref in df.columns and 'R-carre' in df[col_ref].values:
        
        df_principal = df[df[col_ref] != 'R-carre']
        df_r2 = df[df[col_ref] == 'R-carre']
        
        df = pd.concat([df_principal, df_r2], ignore_index=True)

    # DÉFINITION DES STYLES 
    
    # Styles CSS Globaux (En-têtes et Bordures)
    styles_css = [
        dict(selector="th", props=[
            ("background-color", "#2C3E50"), # Bleu nuit
            ("color", "white"),
            ("font-family", "Arial, sans-serif"),
            ("font-weight", "bold"),
            ("text-align", "center"),
            ("border", "1px solid black"),
            ("padding", "10px")
        ]),
        dict(selector="td", props=[
            ("font-family", "Arial, sans-serif"),
            ("border", "1px solid black"),
            ("padding", "8px"),
            ("color", "black"),
            ("text-align", "center")
        ]),
        dict(selector="table", props=[
            ("border-collapse", "collapse"),
            ("border", "2px solid black"),
            ("width", "100%")
        ])
    ]

    # Fonction de style conditionnel (pour la ligne R-carre)
    def style_ligne_speciale(row):
        if row.get(col_ref) == 'R-carre':
            return ['background-color: #FFECB3; font-weight: bold; color: black; border: 1px solid black'] * len(row)
        else:
            return [''] * len(row)

    #  APPLICATION ET AFFICHAGE
    styled_table = (df.style
                    .set_table_styles(styles_css)       # Applique le design global
                    .hide(axis='index')                 # Cache les numéros de ligne
                    .apply(style_ligne_speciale, axis=1)# Applique la couleur sur la dernière ligne
                    .format(precision=2, thousands=" ") # Format des nombres
                   )
    
    display(styled_table)



def regression_iterative(df, target_col, all_feature_cols, country_col, liste_pays, seuil_pvalue=0.05):
    """
    Implémente la stratégie Backward Elimination pour chaque pays.
    """
    
    resultats_globaux = []
    
    print(f"Démarrage de la sélection Stepwise pour {len(liste_pays)} pays...  Pays : coef (p_value) \n")
    for pays in liste_pays:
        # SÉLECTION DES VARIABLES (Backward Elimination)
        
        # 1. Isolation temporaire des données pour le calcul

        df_pays = df[df[country_col] == pays].copy()
        
        cols_utiles = [target_col] + all_feature_cols

        y = df_pays[target_col]
        
        features_candidates = list(all_feature_cols)
        
        while True:
            # S'il y a plus de variables candidates on arrête
            if not features_candidates:
                break
                
            X = df_pays[features_candidates]
            X = sm.add_constant(X) 
            
            # Ajustement du modèle brut
            model = sm.OLS(y, X).fit()
            
            pvalues = model.pvalues
            
            # On cherche la pire p-value (en excluant la constante 'const')
            pvalues_test = pvalues.drop('const', errors='ignore')
            
            if pvalues_test.empty:
                break
                
            pire_pvalue = pvalues_test.max()
            pire_variable = pvalues_test.idxmax()
            
            # Condition de sortie : Si la pire variable est significative, on a fini
            if pire_pvalue <= seuil_pvalue:
                break
            else:
                # Sinon, on la retire de la liste des candidats et on recommence
                features_candidates.remove(pire_variable)
        
        
        # Maintenant qu'on a la liste optimale 'features_candidates' pour ce pays,
        
        dict_resultat = regression_pays(
            df=df,
            feature_cols=features_candidates, # On passe seulement les gagnants
            pays_cible=pays,
            target_col=target_col,
            country_col=country_col
        )

        resultats_globaux.append(dict_resultat)


    df_resultats = pd.DataFrame(resultats_globaux)
   
    if not df_resultats.empty:
        # On pivote pour avoir : Colonnes = Pays, Lignes = Variables
        df_final = df_resultats.set_index('Pays').T
        df_final = df_final.fillna("-")
        
        # Préparation pour l'affichage
        df_final_pour_affichage = df_final.reset_index().rename(columns={'index': 'Variable'})
        
        # Appel de votre fonction d'affichage
        afficher(df_final_pour_affichage)
        
        return df_final 
    else:
        print("Aucun résultat.")
        return pd.DataFrame()


