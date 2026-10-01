"""
fraud_api.py — API REST de détection de fraude (modèle réduit à 6 variables)
=============================================================================
Prérequis : pip install fastapi uvicorn joblib scikit-learn pandas numpy
Lancement : uvicorn fraud_api:app --host 0.0.0.0 --port 8000 --reload
Swagger UI : http://localhost:8000/docs
"""

# --- Imports ---
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import pandas as pd
import numpy as np
import joblib
import json
import logging
from datetime import datetime
from typing import Optional

# --- Configuration du logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("fraud_api.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

#  CHARGEMENT DES ARTEFACTS AU DÉMARRAGE DE L'API

# Les artefacts sont chargés UNE SEULE FOIS au démarrage
# et gardés en mémoire → latence minimale pour chaque requête.

logger.info("Chargement des artefacts du modèle...")
try:
    MODELE  = joblib.load("modele_fraude/modele_fraude_reduit.pkl")

    with open("modele_fraude/colonnes_reduit.json") as f:
        COLONNES_MODELE = json.load(f)

    with open("modele_fraude/metadata_reduit.json") as f:
        METADATA = json.load(f)

    SEUIL   = METADATA["seuil_optimal"]
    VERSION = METADATA.get("version", "2.0.0-reduit")

    logger.info(f" Modèle chargé — version {VERSION} | seuil={SEUIL:.4f}")
    logger.info(f"   Features attendues : {len(COLONNES_MODELE)}")

except FileNotFoundError as e:
    logger.error(f" Fichier introuvable : {e}")
    raise

# Variables du modèle réduit (issues de l'analyse SHAP)
# Les colonnes finales attendues par le modèle sont dans colonnes_reduit.json :
# Account Age Days, Transaction Amount, Transaction Hour, amount_per_account_day,
# Device Used_mobile, Payment Method_credit card
FEATURES_NUM = ["Transaction Amount", "Transaction Hour", "Account Age Days", "amount_per_account_day"]
FEATURES_CAT = ["Payment Method", "Device Used"]

#  MODÈLES PYDANTIC — Validation des entrées/sorties
class TransactionInput(BaseModel):
    """
    Schéma d'entrée : définit et valide les champs d'une transaction.
    Le modèle réduit n'a besoin que de 5 informations.
    FastAPI utilise ce schéma pour générer automatiquement la doc Swagger.
    """
    transaction_id      : Optional[str]  = Field(None, description="Identifiant unique de la transaction")
    transaction_amount  : float          = Field(..., gt=0, description="Montant de la transaction (> 0)")
    payment_method      : str            = Field(..., description="Mode de paiement : bank transfer, debit card, paypal, credit card")
    device_used         : str            = Field(..., description="Appareil : mobile, desktop, tablet")
    account_age_days    : int            = Field(..., ge=0, description="Ancienneté du compte en jours")
    transaction_hour    : int            = Field(..., ge=0, le=23, description="Heure de la transaction (0–23)")

    @field_validator("payment_method")
    @classmethod
    def valider_payment(cls, v):
        valides = {"bank transfer", "debit card", "paypal", "credit card"}
        if v.lower() not in valides:
            raise ValueError(f"Moyen de paiement invalide. Valeurs acceptées : {valides}")
        return v.lower()

    @field_validator("device_used")
    @classmethod
    def valider_device(cls, v):
        valides = {"mobile", "desktop", "tablet"}
        if v.lower() not in valides:
            raise ValueError(f"Appareil invalide. Valeurs acceptées : {valides}")
        return v.lower()


class PredictionOutput(BaseModel):
    """Schéma de sortie : réponse structurée de l'API."""
    transaction_id   : Optional[str]
    score_fraude     : float   = Field(..., description="Probabilité de fraude entre 0 et 1")
    est_frauduleuse  : bool    = Field(..., description="True si score >= seuil")
    niveau_risque    : str     = Field(..., description="FAIBLE | MODÉRÉ | ÉLEVÉ | CRITIQUE")
    seuil_utilise    : float
    timestamp        : str


class StatutAPI(BaseModel):
    """Statut de santé de l'API."""
    statut          : str
    version_modele  : str
    seuil           : float
    n_features      : int
    timestamp       : str


#  FONCTIONS UTILITAIRE

def preprocess(tx: TransactionInput) -> pd.DataFrame:
    """
    Transforme un objet TransactionInput en DataFrame prêt pour la prédiction.
    Doit reproduire exactement le pipeline du notebook pour les 6 variables retenues.
    """
    # Variables brutes
    raw = {
        "Transaction Amount" : tx.transaction_amount,
        "Transaction Hour"   : tx.transaction_hour,
        "Account Age Days"   : tx.account_age_days,
        "Payment Method"     : tx.payment_method,
        "Device Used"        : tx.device_used,
    }

    # Feature engineering
    raw["amount_per_account_day"] = tx.transaction_amount / (tx.account_age_days + 1)

    df_tx = pd.DataFrame([raw])[FEATURES_NUM + FEATURES_CAT]

    # One-hot encoding
    # drop_first=False est OBLIGATOIRE ici : avec une seule ligne, drop_first=True
    # supprimerait la seule modalité présente (les colonnes seraient alors remises à 0).
    # Les colonnes inutiles sont éliminées à l'étape d'alignement ci-dessous.
    df_enc = pd.get_dummies(df_tx, columns=FEATURES_CAT, drop_first=False)

    # Aligner les colonnes sur le modèle (colonnes manquantes = 0, colonnes en trop ignorées)
    for col in COLONNES_MODELE:
        if col not in df_enc.columns:
            df_enc[col] = 0

    return df_enc[COLONNES_MODELE]


def niveau_risque(proba: float, seuil: float) -> str:
    """Classe le risque en 4 niveaux selon la probabilité."""
    if proba < 0.2:        return "FAIBLE"
    elif proba < 0.5:      return "MODÉRÉ"
    elif proba < seuil:    return "ÉLEVÉ"
    else:                  return "CRITIQUE"


#  CRÉATION DE L'APPLICATION FASTAPI


app = FastAPI(
    title=" API Détection de Fraude",
    description=(
        "API REST de scoring de fraude en temps réel. "
        "Envoyez une transaction JSON → recevez un score de risque et une décision.\n\n"
        "**Modèle :** HistGradientBoostingClassifier (sklearn), 6 variables\n\n"
        f"**Version :** {VERSION}"
    ),
    version=VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS : autorise les appels depuis n'importe quelle origine (à restreindre en production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#  ROUTES DE L'API

@app.get("/", tags=["Racine"])
def accueil():
    """Message de bienvenue avec les liens utiles."""
    return {
        "message"   : "🛡️ API Détection de Fraude — opérationnelle",
        "docs"      : "/docs",
        "sante"     : "/health",
        "prediction": "/predict",
        "version"   : VERSION,
    }


@app.get("/health", response_model=StatutAPI, tags=["Santé"])
def sante():
    """
    Endpoint de vérification de santé (*health check*).
    Utilisé par les systèmes de monitoring pour vérifier que l'API est vivante.
    """
    return StatutAPI(
        statut         = "OK",
        version_modele = VERSION,
        seuil          = SEUIL,
        n_features     = len(COLONNES_MODELE),
        timestamp      = datetime.now().isoformat(),
    )

@app.post("/predict", response_model=PredictionOutput, tags=["Prédiction"])
def predire(transaction: TransactionInput):
    """
    **Endpoint principal : prédiction de fraude.**

    Envoie une transaction JSON et reçoit :
    - `score_fraude` : probabilité entre 0 et 1
    - `est_frauduleuse` : True/False
    - `niveau_risque` : FAIBLE | MODÉRÉ | ÉLEVÉ | CRITIQUE
    """
    try:
        # 1. Pré-traitement
        X = preprocess(transaction)

        # 2. Prédiction
        proba  = float(MODELE.predict_proba(X)[0, 1])
        fraude = proba >= SEUIL

        # 3. Logging
        logger.info(
            f"Prédiction | id={transaction.transaction_id} "
            f"| montant={transaction.transaction_amount} "
            f"| score={proba:.4f} | fraude={fraude}"
        )

        return PredictionOutput(
            transaction_id  = transaction.transaction_id,
            score_fraude    = round(proba, 6),
            est_frauduleuse = bool(fraude),
            niveau_risque   = niveau_risque(proba, SEUIL),
            seuil_utilise   = round(SEUIL, 4),
            timestamp       = datetime.now().isoformat(),
        )

    except HTTPException:
        raise  # re-lever les erreurs de validation
    except Exception as e:
        logger.error(f"Erreur interne : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur interne du serveur : {str(e)}")


@app.post("/predict/batch", tags=["Prédiction"])
def predire_batch(transactions: list[TransactionInput]):
    """
    **Prédiction en lot (batch).**
    Envoie une liste de transactions et reçoit une liste de résultats.
    Utile pour le retraitement de fichiers historiques.
    """
    if len(transactions) > 1000:
        raise HTTPException(
            status_code=413,
            detail="Lot trop volumineux : maximum 1 000 transactions par requête."
        )

    resultats = []
    for tx in transactions:
        try:
            X     = preprocess(tx)
            proba = float(MODELE.predict_proba(X)[0, 1])
            resultats.append({
                "transaction_id"  : tx.transaction_id,
                "score_fraude"    : round(proba, 6),
                "est_frauduleuse" : bool(proba >= SEUIL),
                "niveau_risque"   : niveau_risque(proba, SEUIL),
            })
        except Exception as e:
            resultats.append({
                "transaction_id" : tx.transaction_id,
                "erreur"         : str(e)
            })

    n_fraudes = sum(1 for r in resultats if r.get("est_frauduleuse"))
    logger.info(f"Batch | {len(transactions)} transactions | {n_fraudes} fraudes détectées")

    return {
        "n_transactions"  : len(transactions),
        "n_fraudes"       : n_fraudes,
        "taux_fraude"     : round(n_fraudes / len(transactions), 4),
        "resultats"       : resultats,
        "timestamp"       : datetime.now().isoformat(),
    }


#  POINT D'ENTRÉE (pour lancer directement avec python)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fraud_api:app", host="0.0.0.0", port=8000, reload=True)
#uvicorn fraud_api:app --host 0.0.0.0 --port 8000 --reload    
#http://localhost:8000/docs    