import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import prince


# On demande de réinitialiser le style par défaut de seaborn 
sns.reset_orig()

def calculer_acp_pays(data, pays_nom, colonnes_quanti):
    """
    Filtre les données pour un pays et calcule l'ACP sur les variables quantitatives.
    """
    #Filtrage sur le pays
    if isinstance(pays_nom, list):
        subset = data[data['COUNTRY'].isin(pays_nom)].copy()
    else:
        subset = data[data['COUNTRY'] == pays_nom].copy()
    
    # SELECTION:On ne garde que les variables quantitatives
    data_for_pca = subset[colonnes_quanti]

    #Normalisation 
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_for_pca)

    #Calcul de l'ACP
    pca = PCA()
    pca_features = pca.fit_transform(data_scaled)
    
    # On retourne les noms des colonnes quanti pour le cercle des corrélations
    return pca, pca_features, colonnes_quanti



def trace_cercle_et_variance(pca, colonnes_noms, titre_prefixe=""):
    """
    Barplot des variances expliquées par chaque axe et le cercle des corrélations
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 18))

    #Graphique de la variance 
    variance = pca.explained_variance_ratio_
    ax1.bar(range(1, len(variance) + 1), variance, color='skyblue')
    ax1.plot(range(1, len(variance) + 1), np.cumsum(variance), marker='o', color='red')
    ax1.set_title(f"{titre_prefixe} - Variance expliquée", fontsize=14)
    ax1.set_xlabel("Axes")
    ax1.set_ylabel("% de variance")

    #Cercle des corrélations
    pcs = pca.components_
    # On dessine les flèches
    for i, col in enumerate(colonnes_noms):
        ax2.arrow(0, 0, pcs[0, i], pcs[1, i], color='r', alpha=0.5, head_width=0.03)
        # Ajustement du texte pour qu'il ne chevauche pas les flèches
        ax2.text(pcs[0, i]*1.1, pcs[1, i]*1.1, col, color='black', 
                 ha='center', va='center', fontsize=11, fontweight='bold')

    #Dessin cercle unité
    circle = plt.Circle((0,0), 1, color='navy', fill=False, linestyle='--')
    ax2.add_artist(circle)
    
    ax2.set_xlim(-1.2, 1.2)
    ax2.set_ylim(-1.2, 1.2)
    ax2.axhline(0, color='black', lw=1, alpha=0.5)
    ax2.axvline(0, color='black', lw=1, alpha=0.5)
    
    # Titres et labels avec les pourcentages de variance
    ax2.set_title(f"{titre_prefixe} - Cercle des corrélations (Axe 1 & 2)", fontsize=14, pad=10)
    ax2.set_xlabel(f"F1 ({variance[0]:.1%})", fontsize=12)
    ax2.set_ylabel(f"F2 ({variance[1]:.1%})", fontsize=12)
    
    #Pour que le cercle soit vraiment rond:
    ax2.set_aspect('equal')

    plt.tight_layout(pad=4.0)
    plt.show()



def obtenir_matrice_pca(data, pays_nom, colonnes_quanti):
    """
    Renvoie la matrice de covariance.
    """
    # filtrage du pays
    subset = data[data['COUNTRY'] == pays_nom][colonnes_quanti].dropna()
    
    # Normalisation nécessaire à l'acp
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(subset)
    
    #calcul de l'ACP
    pca = PCA()
    pca.fit(data_scaled)
    
    #Extraction de la matrice
    matrice = pca.get_covariance()

    return pd.DataFrame(matrice, index=colonnes_quanti, columns=colonnes_quanti)




    
def simplifier_rsod(x):
    """simplifie les catégories de RSOD_2b (passe de 10 à 4)"""
    if x == 0: return 'Binge-Jamais' # Utilise un tiret simple ici
    elif x in [7, 8, 9]: return 'Binge-Rare'
    elif x in [4, 5, 6]: return 'Binge-Regulier'
    elif x in [1, 2, 3]: return 'Binge-Intensif'
    else: return np.nan

def recoder_f1b(x):
    """
    Recodage spécifique pour f_1b (11 modalités en 4). Regroupe les fréquences par catégories de consommation.
    """
    if x in [1, 2, 3]: return 'Freq-Quotidien'
    elif x in [4, 5, 6]: return 'Freq-Regulier'
    elif x in [7, 8, 9]: return 'Freq-Occasionnel'
    elif x in [10, 11]: return 'Freq-Abstinent'
    else: return np.nan



def tracer_cos2_individus(data, pays_nom, colonnes_quanti, pca, n_top=10):
    # On isole les données du pays
    df_pays = data[data['COUNTRY'] == pays_nom][colonnes_quanti]
    
    # On réduit et centre les données(indispensable pour que les distances soient justes)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df_pays)
    
    # Coordonnées des individus sur les axes
    coords = pca.transform(data_scaled)
    dist_2 = np.sum(data_scaled**2, axis=1)# Distance au carré de chaque individu à l'origine, c'est la somme des carrés des valeurs normalisées de l'individu
    # Calcul du cos2 (coordonnée^2 / distance_totale_2)
    #On évite la division par zéro enajoutant 10^-9 sans biaiser résultat
    cos2 = (coords**2) / (dist_2[:, np.newaxis] + 1e-9)

    # Tracé
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for i in range(2): # On trie par les meilleurs cos2 sur l'axe i
        indices_pos = np.argsort(cos2[:, i])[-n_top:][::-1]
        
        # On récupère les vrais index de 'data' pour ce pays
        vrais_index = df_pays.index[indices_pos].astype(str)
        values = cos2[indices_pos, i]
        
        sns.barplot(x=values, y=vrais_index, ax=axes[i], palette="magma")
        axes[i].set_title(f"{pays_nom} - Top {n_top} Cos2 (Axe {i+1})")
        axes[i].set_xlabel("Qualité de représentation")
        sns.despine(ax=axes[i])

    plt.tight_layout()
    plt.show()
    
    # On retourne l'index du meilleur pour pouvoir l'analyser après
    return df_pays.index[np.argsort(cos2[:, 0])[-1]], df_pays.index[np.argsort(cos2[:, 1])[-1]]




def fait_acm(data, pays_nom, vars_actives):
    """
    Exécute l'ACM sur un pays donné en séparant les variables actives et illustratives.
    """
    # Filtrage sur le pays cible
    subset = data[data['COUNTRY'] == pays_nom].copy()
    # Préparation des données 
    df_actives = subset[vars_actives].astype(str)
    # ACM 
    mca = prince.MCA(
    n_components=3,
    n_iter=3,
    copy=True,
    check_input=True,
    engine='sklearn',
    random_state=42)
    mca = mca.fit(df_actives)
        
    return mca

def plot_var_acm(acm, pays_nom):
    # Récupérer les pourcentages de variance expliquée
    eigenvalues = acm.eigenvalues_
    explained_variance_ratio = eigenvalues / eigenvalues.sum()
    #barplot des % de variance expliquées par axe
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio * 100, alpha=0.7)
    plt.xlabel('Axe')
    plt.ylabel('Pourcentage de variance expliquée (%)')
    plt.title(f'Pourcentage de variance expliquée par chaque axe (ACM {pays_nom})')
    plt.xticks(range(1, len(explained_variance_ratio) + 1))
    for i, v in enumerate(explained_variance_ratio * 100):
        plt.text(i + 1, v + 0.5, f"{v:.1f}%", ha='center')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def tracer_graphe_liaisons(mca, pays_nom):
    """
    Représente la force de liaison de chaque variable avec les deux axes.
    """
    #Extraction des contributions de chaque variable à la construction des axes
    contrib = mca.column_contributions_
    
    # On regroupe les contributions des modalités d'une même variable en une seule valeur par variable(ex: SD_1homme et SD_1femme et SD_1autre)
    contrib['Var'] = [str(c).rsplit('_', 1)[0] for c in contrib.index]
    liaison = contrib.groupby('Var').sum()
    
    # graphique
    plt.figure(figsize=(9, 7))
    
    # On trace les points
    plt.scatter(liaison[0], liaison[1], c='red', s=100, edgecolors='white', linewidth=1.5)
    
    # Ajout des noms des variables
    for i, txt in enumerate(liaison.index):
        plt.annotate(txt, (liaison[0].iloc[i], liaison[1].iloc[i]), 
                     xytext=(7, 7), textcoords='offset points', fontsize=11, fontweight='bold')

    limit = max(liaison[[0, 1]].max()) * 1.1
    plt.xlim(-0.01, limit)
    plt.ylim(-0.01, limit)
    
    plt.axhline(0, color='black', linewidth=1, alpha=0.5)
    plt.axvline(0, color='black', linewidth=1, alpha=0.5)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.title(f"Graphe des liaisons des variables - {pays_nom}", fontsize=13)
    plt.xlabel("Liaison avec l'Axe 1")
    plt.ylabel("Liaison avec l'Axe 2 ")
    
    plt.tight_layout()
    plt.show()

def plot_modalites_acm_simple(data,mca, vars_actives, pays_nom, xlim=None, ylim=None):
    """
    Trace le nuage des modalités
    """
    # Extraction des coordonnées des modalités (Colonnes)
    # Dans prince MCA, les modalités sont les "columns"
    coords = mca.column_coordinates(data[data['COUNTRY'] == pays_nom][vars_actives].astype(str))
    
    plt.figure(figsize=(26, 22))
    
    # On trace chaque modalité
    plt.scatter(coords[0], coords[1], c='red', edgecolors='white', s=60)
    
    # Ajout des labels des variables et modalités
    for i, txt in enumerate(coords.index):
        plt.annotate(txt, (coords.iloc[i, 0], coords.iloc[i, 1]), 
                     xytext=(5, 5), textcoords='offset points', 
                     fontsize=9, alpha=0.8, fontweight='bold')


    plt.axhline(0, color='black', linestyle='--', alpha=0.3)
    plt.axvline(0, color='black', linestyle='--', alpha=0.3)
    
    plt.title(f'Nuage des modalités (ACM) - {pays_nom}', fontsize=14, fontweight='bold')
    plt.xlabel(f'Axe 1 ({mca.percentage_of_variance_[0]:.1f}%)')
    plt.ylabel(f'Axe 2 ({mca.percentage_of_variance_[1]:.1f}%)')
    plt.grid(True, alpha=0.2)
    if xlim: plt.xlim(xlim) #on utilisera ça quand dans un des plots il y a des modalités trop excentrées qui gênent la visualisation
    if ylim: plt.ylim(ylim)
    
    plt.show()


def plot_individus_acm(acm, data, pays_nom, vars_actives):
    # Filtre les données pour le pays
    subset = data[data['COUNTRY'] == pays_nom].copy()

    # Récupérer les coordonnées des individus 
    coords = acm.transform(subset[vars_actives].astype(str))

    # Trace le nuage des individus
    plt.figure(figsize=(10, 8))
    plt.scatter(coords[0], coords[1], alpha=0.5) 
    plt.xlabel('Axe 1')
    plt.ylabel('Axe 2')
    plt.axhline(0, color='black', linestyle='--', alpha=0.3)
    plt.axvline(0, color='black', linestyle='--', alpha=0.3)
    plt.title(f'Nuage des individus (ACM {pays_nom})')
    plt.grid()
    plt.show()

