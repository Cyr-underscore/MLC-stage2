
from typing import Optional, Dict
import torch
import shap
import joblib
import json
import pandas as pd
import os
import socket
import warnings
from sklearn.impute import SimpleImputer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

### --------------- Medical LLM Summary Generator ---------------- ###

# -------------------- MEDICAL LLM --------------------

class MedicalSummaryGenerator:
    """Handles AI-based medical summaries using TinyLlama or bigger llm model"""

    def __init__(self, Big_LLM: bool = False):
        """
        :param Big_LLM: 
            - False → utilise TinyLlama-1.1B (léger, rapide)
            - True  → utilise peut-être phi 3  mini 
        """
        self.Big_LLM = Big_LLM
        self.model_name = None
        self.device = self._setup_device()
        self.model = None
        self.tokenizer = None
        self.load_model()

    # ---------------------------------------------------------------------
    def _setup_device(self) -> str:
        """Detect GPU availability"""
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                gpu_name = torch.cuda.get_device_name(0)
                vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"🎯 GPU detected: {gpu_name} ({vram_gb:.1f}GB VRAM)")
                return "cuda" if vram_gb >= 4 else "cpu"
            except Exception as e:
                print(f"❌ GPU error: {e}")
                return "cpu"
        print("❌ CUDA not available - CPU mode activated")
        return "cpu"

    # ---------------------------------------------------------------------
    def load_model(self):
        """Load models compatible with CUDA 12.6"""
        # Sélection du modèle
        if self.Big_LLM:
            # Essayer Phi-3 d'abord, puis fallbacks si échec
            model_options = [
                "microsoft/Phi-3-mini-4k-instruct",  # Premier choix
                "Qwen/Qwen1.5-1.8B-Chat",           # Fallback 1
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0" # Fallback 2
            ]
        else:
            model_options = ["TinyLlama/TinyLlama-1.1B-Chat-v1.0"]

        # Essayer chaque modèle jusqu'à ce qu'un fonctionne
        for model_attempt in model_options:
            self.model_name = model_attempt
            print(f"🔮 Attempting to load {self.model_name} on {self.device}...")
            
            if self._try_load_single_model():
                print(f"✅ Successfully loaded {self.model_name}")
                return
        
        # Si tous échouent, fallback ultime
        print("❌ All model attempts failed, using ultra-light fallback")
        self._load_ultralight_model()



# ---------------------------------------------------------------------

    def _try_load_single_model(self) -> bool:
        """Try to load a single model, return success status"""
        try:
            # Configuration commune
            common_kwargs = {
                "trust_remote_code": True,  # CRITICAL pour Phi-3
                "low_cpu_mem_usage": True,
            }

            # Chargement du tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=True  # Important ici aussi
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Configuration spécifique au device
            if self.device == "cuda":
                # on utilise 8-bit pour économiser la VRAM
                quant_config = BitsAndBytesConfig(load_in_8bit=True)
                
                model_kwargs = {
                    **common_kwargs,
                    "torch_dtype": torch.float16,
                    "device_map": "auto",
                    "quantization_config": quant_config,
                }
            else:
                model_kwargs = {
                    **common_kwargs,
                    "torch_dtype": torch.float32,
                    "device_map": None,
                }

            # Chargement du modèle
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                **model_kwargs
            )

            # Vérification que le modèle est sur le bon device
            if self.device == "cpu":
                self.model = self.model.to("cpu")
                torch.set_num_threads(min(4, torch.get_num_threads()))

            print(f"💡 Model info: {self.model_name} on {self.device}")
            return True

        except Exception as e:
            print(f"⚠️ Failed to load {self.model_name}: {e}")
            # Nettoyer avant prochaine tentative
            if hasattr(self, 'model') and self.model is not None:
                del self.model
            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                del self.tokenizer
            self.model = None
            self.tokenizer = None
            torch.cuda.empty_cache()
            return False


    # ---------------------------------------------------------------------
    def _load_ultralight_model(self):
        """Fallback ultra-léger"""
        self.model_name = "microsoft/DialoGPT-small"
        print(f"🔄 Fallback to {self.model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            
            if self.device == "cuda":
                self.model = self.model.to("cuda")
            else:
                self.model = self.model.to("cpu")
                torch.set_num_threads(min(4, torch.get_num_threads()))
                
            print("✅ Fallback model loaded")
        except Exception as e:
            print(f"💥 Critical: Even fallback failed: {e}")
            # Dernier recours - modèle minimal
            self._load_minimal_model()

    # ---------------------------------------------------------------------
    def _load_minimal_model(self):
        """Dernier recours - modèle très basique"""
        from transformers import GPT2Tokenizer, GPT2LMHeadModel
        self.model_name = "gpt2"
        print(f"🚨 Emergency fallback to {self.model_name}")
        
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.model.to("cpu")
        print("✅ Minimal model loaded")

    # ---------------------------------------------------------------------
    def generate_medical_summary(self, patient_data: Dict) -> str:
        import time
        import threading

        def generation_with_timeout():
            nonlocal result, generation_error, completed

            try:
                # English prompt
                prompt = self.build_english_prompt(patient_data)

                print("=" * 80)
                print("📝 PROMPT ENGLISH:")
                print(prompt)
                print("=" * 80)

                # Tokenization
                max_len = 4096 if self.Big_LLM else 1024
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    max_length=max_len,
                    truncation=True,
                    padding=True
                )

                device = next(self.model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=512 if self.Big_LLM else 400,
                        temperature=0.7,
                        do_sample=True,
                        top_p=0.9,
                        top_k=50,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        repetition_penalty=1.1,
                        no_repeat_ngram_size=2
                    )

                # Decode the generated text
                full_generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                print("📄 FULL GENERATED TEXT:")
                print(full_generated_text)

                # Extract only the part after the prompt
                raw_result = full_generated_text[len(prompt):].strip()

                if raw_result:
                    last_period = raw_result.rfind('.')
                    if last_period != -1:
                        raw_result = raw_result[:last_period + 1]

                    if len(raw_result.split()) < 3:
                        raw_result = "Unable to generate comprehensive summary."

                result = self._format_summary_html(raw_result)
                print(f"🎯 RESULT: '{raw_result}'")

                completed = True

            except Exception as e:
                generation_error = str(e)
                print(f"💥 ERROR: {generation_error}")
                completed = True

        result = ""
        generation_error = None
        completed = False

        print("🚀 STARTING MEDICAL SUMMARY GENERATION")
        start_time = time.time()

        thread = threading.Thread(target=generation_with_timeout)
        thread.daemon = True
        thread.start()
        thread.join(timeout=260.0 if self.Big_LLM else 120.0)

        generation_time = time.time() - start_time
        print(f"⏱️  Total time: {generation_time:.1f}s")

        if not completed:
            return self._format_error_html("Timeout - Model took too long to respond")

        if generation_error:
            return self._format_error_html(f"Error: {generation_error}")

        return result if result else "No AI. Unability to generate summary."

    # ---------------------------------------------------------------------
    def _format_summary_html(self, summary_text: str) -> str:
        """Format the AI summary as plain text"""
        if not summary_text or summary_text == "Unable to generate comprehensive summary.":
            return self._format_error_html("No summary generated by the AI model")

        return summary_text.strip()

    def _format_error_html(self, error_message: str) -> str:
        """Format error messages as plain text"""
        return f"Error: {error_message}"


    # ---------------------------------------------------------------------
    def build_english_prompt(self, patient_data: Dict) -> str:
        variable_mapping = {
            "age": "Age",
            "male": "Gender (0=F, 1=M)",
            "days_from_diag": "Days since diagnosis",
            "diag_DC18": "Colorectal cancer",
            "diag_DC34": "Lung cancer",
            "diag_DC50": "Breast cancer",
            "diag_DC79": "Metastases",
            "comorb_met": "Metastatic disease",
            "comorb_CKD": "Chronic kidney disease",
            "comorb_COPD": "COPD",
            "comorb_diabetes1": "Diabetes",
            "blood_test_NPU01349": "Hemoglobin",
            "blood_test_NPU01685": "Creatinine",
            "blood_test_NPU03011": "CRP",
            "vitals_bmi": "BMI",
            "side_effect_fatigue": "Fatigue",
            "side_effect_pain": "Pain",
            "side_effect_anemia": "Anemia",
            "drug_prn0_L01AA01": "Chemotherapy",
            "RT_BWGC1": "Radiotherapy",
            "surgery_KGEA00": "Surgery"
        }

        data_items = []
        for key, display_name in variable_mapping.items():
            value = patient_data.get(key)
            if value is not None and value != '':
                data_items.append(f"{display_name}: {value}")

        formatted_data = " | ".join(data_items)

        prompt = f"""Context: You are a medical AI assistant in oncology. Analyze patient data and provide a concise clinical summary.

Guidelines:
- 0=No/Absent, 1=Yes/Present
- BMI <18.5=underweight, CRP>1=inflammation
- Focus on cancer status, key findings, and recommendations
- Do not specify your name or role

Patient data: {formatted_data}

Medical summary: The patient is a"""
        return prompt