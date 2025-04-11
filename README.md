# Fine-tuning VITS pour le dialecte arabe tunisien

Ce projet permet d’effectuer un fine-tuning du modèle MMS-TTS sur un dataset de type Hugging Face (`Elyadata/TunArTTS`) pour générer une voix en arabe tunisien.

---

## 🛠 Étape 1 : Conversion du modèle original

Avant le fine-tuning, il est nécessaire de convertir les checkpoints du modèle original MMS :

```bash
python convert_original_discriminator_checkpoint.py \
  --language_code ara \
  --pytorch_dump_folder_path arabic-mms-checkpoint \
  --push_to_hub tonusername/arabic-mms-tts
```

👉 **Pourquoi ?**  
Le modèle original n'est pas directement exploitable. Cette conversion génère une version compatible avec Hugging Face et PyTorch, tout en permettant un éventuel push vers le hub.

---

## ⚙️ Étape 2 : Préparer la configuration du dataset

Dans le fichier de configuration JSON, plusieurs **attributs sont essentiels** pour que le dataset soit correctement utilisé pendant l'entraînement :

| Attribut | Description |
|----------|-------------|
| `dataset_name` | Nom du dataset sur Hugging Face (ex. `"Elyadata/TunArTTS"`) |
| `dataset_config_name` | Configuration interne du dataset (souvent `"default"`) |
| `audio_column_name` | Nom de la colonne contenant les fichiers audio (ex. `"audio"`) |
| `text_column_name` | Nom de la colonne contenant les transcriptions (ex. `"tgt_text_without_diacritization"`) |
| `train_split_name` | Nom du split utilisé pour l'entraînement (ex. `"train"`) |
| `eval_split_name` | Nom du split utilisé pour l’évaluation (peut être `"train"` si un seul split disponible) |

📝 **Remarque :**  
Ces noms doivent **correspondre exactement** à ceux présents dans le dataset sur Hugging Face. Tu peux les vérifier via [le dataset en ligne](https://huggingface.co/datasets/Elyadata/TunArTTS) ou via la fonction `load_dataset`.

---

## 🚀 Étape 3 : Lancer le fine-tuning

Une fois le fichier de config prêt, tu peux lancer l’entraînement avec :

```bash
accelerate launch run_vits_finetuning.py training_config_arabic.json
```

---

## 📌 Résumé

- Convertir le modèle MMS original avec `convert_original_discriminator_checkpoint.py`
- Créer un fichier de configuration JSON avec les bons attributs liés au dataset
- Lancer le fine-tuning avec `accelerate`

---
