# 🧠 Analyse Bimodale de la Santé Mentale Étudiante

## 🌟 Présentation du Projet
Ce projet de Data Science vise à identifier et prédire les profils de santé mentale au sein de la communauté étudiante.
L'approche est **bimodale** : elle croise des **données quantitatives** (scores cliniques PHQ-9 et GAD-7) et des **données qualitatives** (analyse du discours via le NLP).

L'objectif final est de fournir un outil d'aide à la décision pour orienter les étudiants vers des ressources de soutien adaptées.

---

## 🛠️ Pipeline Technique

### 1. Collecte & Prétraitement
- **Source :** Enquête bilingue (FR/EN) auprès de 286 étudiants.
- **Ingénierie des données :** Harmonisation bilingue et calcul du Score de Détresse Psychologique (SDP).

### 2. Le Modèle "Pont" (NLP)
- **Entraînement :** Modèle de classification entraîné sur un large corpus **Reddit** pour détecter les signaux de dépression.
- **Application :** Génération d'un score de probabilité de risque basé sur les témoignages libres des étudiants.

### 3. Clustering (Apprentissage Non-Supervisé)
- **Algorithme :** K-Means.
- **Résultat :** Identification de **3 profils types** :
    - **Profil Robuste :** Haute résilience, fort soutien social.
    - **Profil Fragilisé :** Vulnérabilité modérée, isolement naissant.
    - **Profil à Risque Élevé :** Détresse sévère, fort stress financier et académique.

### 4. Classification (Apprentissage Supervisé)
- **Algorithme :** Random Forest.
- **Performance :** **85.11% d'accuracy** dans la prédiction des profils.
- **Interprétabilité :** Analyse de la *Feature Importance* révélant le **Soutien des Pairs** et l'**Auto-Efficacité** comme les principaux leviers de résilience.

---

## 🚀 Déploiement
Le projet inclut une application interactive réalisée avec **Streamlit**. Elle permet de :
1. Saisir ses scores cliniques et ses ressentis textuels.
2. Obtenir une prédiction immédiate de son profil de risque.
3. Recevoir des recommandations basées sur les facteurs de protection identifiés.

---


## 👩‍💻 Auteur
**Maimouna Oiga** Étudiante en Master Science et Ingénierie des Données.  
*Projet réalisé avec la volonté de mettre la Data Science au service du bien-être communautaire.*
