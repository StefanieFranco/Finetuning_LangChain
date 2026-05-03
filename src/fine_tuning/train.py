import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_from_disk

def run_training():
    # 1. Configurações de Quantização (4-bit para eficiência de VRAM)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # 2. Carregar Modelo Base e Tokenizer (Llama-3-8B ou similar sugerido)
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct" 
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # 3. Preparação para PEFT (LoRA)
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=16, # Rank: Maior = mais parâmetros treináveis
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], # Foco nas camadas de atenção
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)

    # 4. Carregar Dataset Processado (Saída do seu script anterior)
    dataset = load_from_disk("data/processed/medical_finetuning_dataset")

    # 5. Argumentos de Treino (Focado em reprodutibilidade acadêmica)
    training_args = TrainingArguments(
        output_dir="./models/llama-medical-ft",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        max_steps=100, # Ajustar conforme o tamanho do dataset
        fp16=True,
        optim="paged_adamw_32bit",
        report_to="tensorboard", # Essencial para o relatório técnico
        push_to_hub=False
    )

    # 6. Trainer (SFT - Supervised Fine-tuning)
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        dataset_text_field="text", # Campo formatado no preprocess
        max_seq_length=1024,
        tokenizer=tokenizer,
        args=training_args,
    )

    # Iniciar Treinamento
    print("Iniciando Fine-tuning Médico...")
    trainer.train()

    # 7. Salvar o Adaptador LoRA
    trainer.model.save_pretrained("./models/llama-medical-adapter")
    print("Treino concluído e adaptador salvo!")

if __name__ == "__main__":
    run_training()