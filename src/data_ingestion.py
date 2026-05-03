import os
import json
from datasets import load_dataset

def setup_data_folders():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

def ingest_public_datasets():
    """Baixa e salva os dados brutos sugeridos[cite: 1]."""
    print("Baixando PubMedQA...")
    # Salvamos como CSV ou JSON para ter uma cópia física no /raw
    pubmed = load_dataset("pubmed_qa", "pqa_labeled", split="train")
    pubmed.to_csv("data/raw/pubmedqa_raw.csv")
    
    print("Baixando MedQuAD...")
    medquad = load_dataset("keivalya/MedQuAD-MedicalQnA", split="train")
    medquad.to_json("data/raw/medquad_raw.json")

def create_internal_mock_data():
    """Cria o dataset de protocolos internos exigido[cite: 1]."""
    internal_protocols = [
        {
            "id": "PROT-001",
            "titulo": "Protocolo de Antibioticoterapia - Hospital FIAP",
            "conteudo": "Para infecções respiratórias leves, utilizar Amoxicilina 500mg..."
        },
        {
            "id": "LAUDO-001",
            "titulo": "Modelo de Laudo Cardiológico",
            "conteudo": "Paciente apresenta ritmo sinusal, sem evidências de isquemia..."
        }
    ]
    
    with open("data/raw/internal_protocols.json", "w", encoding="utf-8") as f:
        json.dump(internal_protocols, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    setup_data_folders()
    ingest_public_datasets()
    create_internal_mock_data()
    print("Ingestão concluída. Dados salvos em ./data/raw[cite: 1]")