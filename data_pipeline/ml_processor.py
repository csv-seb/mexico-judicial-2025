"""
Judicial Elections 2025 - Semantic Matching Pipeline
Author: Sebastian Méndez Ramírez
Description: This script processes raw electoral data, applies Natural Language Processing (NLP) 
to extract narrative pillars using Latent Dirichlet Allocation (LDA), calculates credential scores, 
and exports a lightweight JSON file for the frontend semantic search engine.
"""

import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def clean_text_fields(val):
    """Removes administrative noise and handles missing data."""
    if pd.isna(val) or str(val).lower() in ["no proporcionó", "no proporciono", "¡conóceles!", "sin información"]:
        return ""
    return str(val).strip()

def process_data(file_path):
    print("Loading data...")
    df = pd.read_csv(file_path)

    # 1. CLEANING THE DATA
    print("Cleaning text fields...")
    df = df[df['ESTATUS'] == 'Publicado'].copy()
    text_cols = ['MOTIVO_CARGO_PUBLICO', 'VISION_FUNCION_JURISDICCIONAL', 
                 'VISION_IMPARTICION_JUSTICIA', 'PROPUESTA_1', 'PROPUESTA_2', 'PROPUESTA_3']

    for col in text_cols:
        df[col] = df[col].apply(clean_text_fields)

    # Create a single corpus per candidate for the Machine Learning model
    df['corpus'] = df[text_cols].apply(lambda x: ' '.join(x), axis=1)

    # 2. NLP & TOPIC MODELING (LDA)
    print("Running Machine Learning Pipeline (TF-IDF & LDA)...")
    
    # Custom stopwords specific to the Mexican legal domain
    judicial_stopwords = [
        'que', 'los', 'del', 'para', 'con', 'las', 'por', 'una', 'este', 'su', 'al', 'en', 'la', 'de',
        'justicia', 'judicial', 'derecho', 'mexico', 'nacional', 'candidato', 'ley', 'legal', 
        'constitucional', 'proceso', 'electoral', 'persona', 'personas', 'ser', 'sus', 'como', 
        'función', 'impartición', 'caso', 'casos', 'través', 'poder', 'cada', 'estos', 'esta', 'se', 'ha', 'lo'
    ]

    # Vectorize the text using N-Grams (2 to 3 words)
    tfidf = TfidfVectorizer(max_df=0.7, min_df=5, stop_words=judicial_stopwords, ngram_range=(2, 3))
    dtm = tfidf.fit_transform(df['corpus'].apply(lambda x: x.lower()))

    # Run Latent Dirichlet Allocation to find 5 distinct Political/Narrative Themes
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    topic_results = lda.fit_transform(dtm)

    # Map the mathematical topics to human-readable labels
    theme_map = {
        0: "Formalista / Constitucional",
        1: "Defensoría Social / Derechos",
        2: "Reforma / Lenguaje Claro",
        3: "Equidad / Anti-Privilegios",
        4: "Eficiencia / Acceso Digital"
    }

    # Assign the dominant pillar to each candidate
    df['pillar'] = [theme_map[i] for i in topic_results.argmax(axis=1)]
    
    # Flag candidates who provided zero text
    df.loc[df['corpus'].str.strip() == "", 'pillar'] = "Sin información"

    # 3. FEATURE ENGINEERING: CREDENTIAL SCORE
    print("Calculating Credential Scores...")
    edu_score = {
        "Doctorado": 10,
        "Postdoctorado": 12,
        "Maestría": 7,
        "Especialidad": 5,
        "Licenciatura": 3,
        "Pasante": 1,
        "Concluido": 3 
    }
    df['cred_score'] = df['ESCOLARIDAD'].map(edu_score).fillna(0)

    # 4. JSON SERIALIZATION
    print("Exporting to JSON for Frontend Deployment...")
    candidates_list = []
    for idx, row in df.iterrows():
        candidates_list.append({
            "name": row['NOMBRE_CANDIDATO'],
            "role": row['CARGO'],
            "state": row['ENTIDAD'],
            "gender": row['SEXO'],
            "education": row['ESCOLARIDAD'],
            "power": row['PODER_POSTULA'],
            "motivo": row['MOTIVO_CARGO_PUBLICO'],
            "vision": row['VISION_FUNCION_JURISDICCIONAL'],
            "propuesta": row['PROPUESTA_1'],
            "pillar": row['pillar'],
            "score": int(row['cred_score'])
        })

    # Calculate global stats for the frontend dashboard
    stats = {
        "total": len(df),
        "gender": df['SEXO'].value_counts().to_dict(),
        "education": df['ESCOLARIDAD'].value_counts().to_dict(),
        "power": df['PODER_POSTULA'].value_counts().head(10).to_dict()
    }

    # Save the file
    with open('../candidates_data.json', 'w', encoding='utf-8') as f:
        json.dump({"stats": stats, "candidates": candidates_list}, f, ensure_ascii=False, indent=2)

    print("Pipeline Complete! Created 'candidates_data.json'")

if __name__ == "__main__":
    process_data('baseDatosCandidatos.csv')
