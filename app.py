import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
from groq import Groq
from transformers import AutoTokenizer, AutoModel
from faiss import read_index, write_index, IndexFlatL2
from dotenv import load_dotenv
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from stop_words import get_stop_words
import faiss
import pickle


prix_min = 0.8 # exemple
prix_max = 5000.0  # exemple

load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
EMBEDDING_MODEL = "BAAI/bge-m3"
FAISS_INDEX_PATH = "models/faiss_index.index"
EMBEDDINGS_PATH = "models/embeddings.npy"
JSON_DATA_PATH = "data/data.json"
SCALER_PATH        = "models/scaler.pkl"
CLUSTER_MODEL_PATH = "models/modele_clustering.pkl"

# Initialiser le client Groq
client = Groq(api_key="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")
clustering_model = joblib.load(CLUSTER_MODEL_PATH)
scaler_2 = joblib.load(SCALER_PATH)
# Charger le modèle d'embedding
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
model = AutoModel.from_pretrained(EMBEDDING_MODEL)

# Variable globale pour stocker les données
document_data = []



with open('models\model_components.pkl', 'rb') as f:
    components = pickle.load(f)
    
tfidf = components['tfidf']
tfidf_matrix = components['tfidf_matrix']
scaler = components['scaler']
df = components['df']

# Precompute similarity matrix
similarity_text = cosine_similarity(tfidf_matrix)

def recommander_produits(index_produit: int, top_n: int = 5) -> pd.DataFrame:
    sim = similarity_text[index_produit]
    diff_prix = np.abs(df['Prix_normalise'] - df.at[index_produit, 'Prix_normalise'])
    score_prix = 1 - diff_prix
    score_note = df['Note_normalisee']
    score_final = 0.6 * sim + 0.2 * score_prix + 0.2 * score_note

    df_scores = df.copy()
    df_scores['score_final'] = score_final
    recommandations = (
        df_scores
        .drop(index=index_produit)
        .sort_values(by='score_final', ascending=False)
        .head(top_n)
    )
    return recommandations[['Nom du produit', 'Prix', 'Note', 'score_final']]


def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

def initialize_rag():
    global document_data

    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(EMBEDDINGS_PATH):
        index = read_index(FAISS_INDEX_PATH)
        embeddings = np.load(EMBEDDINGS_PATH)
        with open(JSON_DATA_PATH) as f:
            document_data = json.load(f)
        return index, embeddings

    # Charger et préparer les données
    with open(JSON_DATA_PATH) as f:
        document_data = json.load(f)
    
    # Créer les textes pour l'embedding
    texts = [
        f"Sujet: {item['sujet']}\nQuestion: {item['contenu']['question']}\nRéponse: {item['contenu']['reponse']}"
        for item in document_data
    ]
    
    # Générer les embeddings
    embeddings = np.vstack([get_embedding(text) for text in texts])
    
    # Créer et sauvegarder l'index FAISS
    index = IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    
    os.makedirs("models", exist_ok=True)
    write_index(index, FAISS_INDEX_PATH)
    np.save(EMBEDDINGS_PATH, embeddings)
    
    return index, embeddings

# Initialiser RAG au démarrage
faiss_index, document_embeddings = initialize_rag()

def rag_retrieval(query, k=3):
    query_embedding = get_embedding(query)
    distances, indices = faiss_index.search(query_embedding, k)
    return [document_data[i] for i in indices[0]]

def generate_response(prompt, context):
    # Formater le contexte pour le LLM
    context_str = "\n\n".join([
        f"Question: {item['contenu']['question']}\nRéponse: {item['contenu']['reponse']}"
        for item in context
    ])
    
    system_prompt = f"""Vous êtes un chatbot de Monoprix France. 
Ne donnez pas de réponses hors contexte et essayez de vous limiter strictement au contexte fourni.

Informations disponibles :
{context_str}

Question du client : {prompt}"""
    
    completion = client.chat.completions.create(
        model="Llama3-70b-8192",
        messages=[
            {
                "role": "system", 
                "content": system_prompt
            },
            {
                "role": "user", 
                "content": "Réponds de manière concise et précise en français. Si l'information n'existe pas dans le contexte, dis simplement que vous ne pouvez pas répondre."
            }
        ],
        temperature=0.3
    )
    
    return completion.choices[0].message.content

@app.route("/")
def home():
    return render_template("index.html")   # Ton site principal

@app.route("/chatbot")
def chatbot():
    return render_template("chat.html")   # L'interface du chatbot

@app.route("/chat", methods=["POST"])
def chat():
    if request.method == "POST":
        user_message = request.json["message"]
        
        try:
            # Récupération de contexte
            context = rag_retrieval(user_message)
            
            # Génération de réponse
            response = generate_response(user_message, context)
            
            return jsonify({"response": response})
        except Exception as e:
            return jsonify({"response": f"Une erreur est survenue : {str(e)}"})
    else:
        return Response(status=405)  # Method Not Allowed
    
    
    
@app.route("/predict_cluster", methods=["POST"])
def predict_cluster():
    data = request.get_json()
    try:
        Note = float(data["Note"])
        Prix = float(data["Prix"])
    except (KeyError, ValueError):
        return jsonify({"error": "Paramètres manquants ou invalides."}), 400

    # 1. Min-max scaling de Prix
    prix_scaled = (Prix - prix_min) / (prix_max - prix_min)

    # 2. Préparer X
    X_raw = np.array([[Note, prix_scaled]])

    # 3. Standardisation
    X_scaled = scaler_2.transform(X_raw)

    # 4. Prédiction
    cluster = clustering_model.predict(X_scaled)[0]

    return jsonify({"cluster": int(cluster)})

@app.route('/recommendation')
def recommendation_page():
    return render_template('rec.html')  # Plus besoin de passer la liste des produits

@app.route("/recom", methods=["POST"])
def recommend():
    try:
        data = request.get_json()
        product_name = data['product_name'].strip().lower()
        top_n = int(data.get('top_n', 5))
        
        # Trouver le produit le plus similaire
        new_tfidf = tfidf.transform([product_name])
        sim_scores = cosine_similarity(new_tfidf, tfidf_matrix).flatten()
        best_match_idx = np.argmax(sim_scores)
        max_similarity = sim_scores[best_match_idx]

        if max_similarity < 0.2:  # Seuil de similarité minimal
            return jsonify({
                'status': 'error',
                'message': 'Aucun produit correspondant trouvé'
            })
        
        recommendations = recommander_produits(best_match_idx, top_n)
        return jsonify({
            'status': 'success',
            'input_product': df.iloc[best_match_idx]['Nom du produit'],
            'recommendations': recommendations.to_dict('records')
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

if __name__ == "__main__":
    app.run(debug=True)