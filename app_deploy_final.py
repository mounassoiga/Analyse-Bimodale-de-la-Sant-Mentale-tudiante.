
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. CHARGEMENT DU MODÈLE FINAL ---
# Assurez-vous que le fichier 'bridge_model_final.pkl' est présent
try:
    bridge_model_final = joblib.load('bridge_model_final.pkl')
except FileNotFoundError:
    st.error("Erreur : Le fichier 'bridge_model_final.pkl' est introuvable. Veuillez d'abord le sauvegarder (joblib.dump).")
    st.stop()

# --- 2. FONCTION D'INFÉRENCE NLP (SIMULATION) ---
def run_nlp_inference(reddit_text):
    """
    Simule la conversion du texte en Prob_Depression_Reddit_Score [0, 1].
    
    Pour la démonstration, nous utilisons une règle simple sur la longueur et la tonalité du texte.
    CETTE FONCTION DOIT ÊTRE REMPLACÉE PAR VOTRE PIPELINE NLP RÉEL EN PRODUCTION.
    """
    text_lower = reddit_text.lower()
    score = 0.0
    
    # Simulation basée sur des mots clés de détresse pour créer un score [0, 1]
    distress_keywords = ['stress', 'anxieté', 'dépression', 'seul', 'peur', 'épuisé', 'aide']
    for keyword in distress_keywords:
        if keyword in text_lower:
            score += 0.15
            
    # Ajustement pour la longueur (un texte long et négatif est plus significatif)
    score += len(text_lower) / 500.0 # Ajout d'une petite variable basée sur la longueur
    
    return min(1.0, score) # S'assurer que le score est <= 1.0


# --- 3. FONCTION DE PRÉDICTION ET D'INTERPRÉTATION (Stratégie Opérationnelle K=2) ---
def predict_and_interpret(features_vector):
    """
    Prédit le cluster (K=3) et interprète le résultat selon la stratégie opérationnelle (Risque vs Non-Risque).
    """
    
    # 8 features utilisées pour l'entraînement du modèle (Accuracy 0.8511)
    feature_columns = [
        'Auto_Efficacite', 
        'Soutien_Familial', 
        'Stress_Academique', 
        'Stress_Financier', 
        'Performance_Academique', 
        'Soutien_Pairs',
        'Prob_Depression_Reddit_Score',
        'SDP_Détresse_Psychologique'
         
    ]
    
    X_new = pd.DataFrame([features_vector], columns=feature_columns)
    predicted_cluster = bridge_model_final.predict(X_new)[0]
    
    # Interprétation basée sur la Matrice de Confusion : Fusion du risque (0 et 2)
    
    if predicted_cluster in [0, 2]:
        # Cluster 2 (Surmené Alarmant) et Cluster 0 (Fragilisé) sont fusionnés ici, 
        # car le modèle les classe de manière similaire (voir Matrice de Confusion).
        
        if predicted_cluster == 2:
            niveau = "Risque ÉLEVÉ (Profil Alarmant)"
            couleur = st.error
            message = "L'étudiant est dans la catégorie de **risque le plus élevé**, caractérisée par une forte détresse psychosociale, confirmée par des indicateurs structuraux et l'analyse textuelle. **Intervention Immédiate Recommandée**."
        else: # predicted_cluster == 0
            niveau = "Risque MODÉRÉ/ÉLEVÉ (Profil Fragilisé)"
            couleur = st.warning
            message = "L'étudiant est classé dans la catégorie 'Fragilisé'. Bien que moins intense que le profil Alarmant, le modèle le place en alerte. **Suivi proactif et renforcement du soutien nécessaires**."
            
        return niveau, message, couleur, predicted_cluster
        
    elif predicted_cluster == 1:
        niveau = "Faible Risque (Profil Robuste)"
        couleur = st.success
        message = "L'étudiant présente un **faible risque de détresse** et des facteurs de protection solides (Auto-Efficacité, Soutien). Un suivi standard est suffisant."
        return niveau, message, couleur,predicted_cluster
        
    else:
        return "Indéterminé", "Erreur de classification.", st.info


# --- 4. INTERFACE STREAMLIT ---
st.title("🤖 Système de Prédiction Psychosociale Bimodale")
st.markdown("### Démonstration de l'Outil d'Aide à la Décision")

st.header("1. Entrée des Scores d'Enquête")
st.caption("Échelles adaptées à votre enquête (ex: 1 à 10 pour les facteurs structurels, 0 à 27 pour le SDP).")

col1, col2 = st.columns(2)

with col1:
    auto_efficacite = st.slider("Auto-Efficacité", min_value=1, max_value=10, value=3)
    soutien_familial = st.slider("Soutien Familial", min_value=1, max_value=10, value=4)
    soutien_pairs = st.slider("Soutien des Pairs/camarades/collégues", min_value=1, max_value=10, value=3)
    stress_academique = st.slider("Stress Académique", min_value=1, max_value=10, value=3)
    
with col2:
    stress_financier = st.slider("Stress Financier", min_value=1, max_value=10, value=4)
    performance_academique = st.slider("Performance Académique (GPA)", min_value=1, max_value=4, value=3)
    # SDP_Détresse_Psychologique
    sdp_score = st.slider("SDP (Détresse Psychologique/Dépression/Anxieté)", min_value=0, max_value=30, value=20)


st.header("2. Entrée des Données Textuelles pour l'Inférence NLP")
reddit_text = st.text_area(
    "Texte Libre de l'Étudiant (Ex. : Publication sur un forum)", 
    "Je suis très stressé par mes examens et je n'arrive pas à me concentrer. Je me sens seul en ce moment, c'est vraiment difficile.",
    height=150
)

if st.button("Analyser et Prédire le Profil"):
    
    # 1. Inférence NLP
    nlp_prob_score = run_nlp_inference(reddit_text)
    st.info(f"Score de Probabilité (NLP Simulé) généré : {nlp_prob_score:.2f}")

    # 2. Construction du vecteur final (8 features)
    final_features_vector = [
        auto_efficacite,
        soutien_familial,
        stress_academique,
        stress_financier,
        performance_academique,
        soutien_pairs,
        sdp_score,
        nlp_prob_score
    ]
    
    # 3. Prédiction et Interprétation
    niveau, message,display_func, predicted_cluster = predict_and_interpret(final_features_vector)
    st.subheader("🎉 Résultat Opérationnel du Modèle Bimodal")
    
    display_func(f"Catégorie de Risque Prédite : **{niveau}**")
    st.markdown("---")
    st.markdown("**Interprétation Détaillée :**")
    st.markdown(message)
    st.markdown(f"*(Le modèle a prédit le Cluster {predicted_cluster} de la classification initiale.)*")