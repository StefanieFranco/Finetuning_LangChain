import re
import pandas as pd
from datasets import Dataset

class MedicalDataProcessor:
    def __init__(self):
        # Regex simples para exemplificação (em prod usaríamos instâncias de Spacy ou Presidio)
        self.pii_patterns = {
            "NOME": r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)+\b",
            "CPF": r"\d{3}\.\d{3}\.\d{3}-\d{2}",
            "DATA": r"\d{2}/\d{2}/\d{4}",
            "EMAIL": r"[\w\.-]+@[\w\.-]+\.\w+"
        }

    def anonymize_text(self, text):
        """Remove dados sensíveis do texto médico."""
        for label, pattern in self.pii_patterns.items():
            text = re.sub(pattern, f"[{label}_ANONIMIZADO]", text)
        return text

    def format_instruction(self, example):
        """Formata para o padrão de instrução do Llama."""
        # Padrão Llama 2/3: <s>[INST] Prompt [/INST] Resposta </s>
        prompt = f"<s>[INST] {example['pergunta']} [/INST] {example['resposta']} </s>"
        return {"text": prompt}

    def process_pipeline(self, df):
        """Executa a curadoria completa."""
        # 1. Limpeza e Anonimização
        df['pergunta'] = df['pergunta'].apply(self.anonymize_text)
        df['resposta'] = df['resposta'].apply(self.anonymize_text)
        
        # 2. Conversão para Hugging Face Dataset
        dataset = Dataset.from_pandas(df)
        
        # 3. Mapeamento para formato de instrução
        dataset = dataset.map(self.format_instruction)
        
        return dataset

# Exemplo de uso com dados sintéticos (exigência do projeto)[cite: 1]
data = {
    "pergunta": ["O paciente João Silva, portador do CPF 123.456.789-00, apresenta febre.", "Qual o protocolo para diabetes?"],
    "resposta": ["O paciente João Silva deve tomar antitérmico.", "Seguir o protocolo interno Hopsital_FIAP_2026."]
}

processor = MedicalDataProcessor()
df = pd.DataFrame(data)
clean_dataset = processor.process_pipeline(df)

# Salvar para o treino
clean_dataset.save_to_disk("data/processed/medical_finetuning_dataset")
print("Dataset anonimizado e formatado com sucesso![cite: 1]")