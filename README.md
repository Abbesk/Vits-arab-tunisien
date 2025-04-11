# Fine-tuning VITS pour le dialecte arabe tunisien

Ce projet permet d’effectuer un fine-tuning du modèle MMS-TTS sur un dataset de type Hugging Face (`Elyadata/TunArTTS`) pour générer une voix en arabe tunisien.

---

## 🛠 Étape 1 : Conversion du modèle original

Avant le fine-tuning, il est nécessaire de convertir les checkpoints du modèle original MMS :

```bash
python convert_original_discriminator_checkpoint.py \
  --language_code ara \
  --pytorch_dump_folder_path arabic-mms-checkpoint \
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
Ces noms doivent **correspondre exactement** à ceux présents dans le dataset sur Hugging Face. Vous pouvez les vérifier via [le dataset en ligne](https://huggingface.co/datasets/Elyadata/TunArTTS) ou via la fonction `load_dataset`.

---

## 🔄 Étape 3 : Télécharger le dataset localement (optionnel)

Si vous souhaitez télécharger le dataset localement plutôt que d'y accéder en ligne, vous pouvez le faire en utilisant la commande suivante :

```bash
from datasets import load_dataset
dataset = load_dataset("Elyadata/TunArTTS")
```

Si vous optez pour un chargement local du dataset, **le fichier de configuration doit être modifié**. En effet, au lieu de spécifier simplement le nom du dataset sur Hugging Face dans le champ `dataset_name`, vous devez spécifier le chemin local où le dataset est stocké, comme ceci :

```json
"dataset_name": "/chemin/vers/votre/dataset",
```

Cela permet à votre script de charger le dataset localement sans tenter de le récupérer en ligne. Assurez-vous également que les colonnes (`audio_column_name`, `text_column_name`) sont correctement définies pour le chemin local.

---

## 📝 Vérification du vocabulaire

Il est essentiel de vérifier que le vocabulaire du dataset correspond au vocabulaire avec lequel le modèle a été entraîné. Pour cela, après avoir exécuté la commande de conversion du modèle, vous trouverez un fichier **`vocab.json`** dans le répertoire où le modèle a été sauvegardé (`arabic-mms-checkpoint`). Ce fichier contient le vocabulaire utilisé pour l'entraînement du modèle.

Assurez-vous que votre dataset utilise un vocabulaire compatible avec celui contenu dans ce fichier afin de garantir une correspondance correcte lors du fine-tuning.

---

## 🚀 Étape 4 : Lancer le fine-tuning

Une fois le fichier de config prêt et le vocabulaire vérifié, vous pouvez lancer l’entraînement avec :

```bash
accelerate launch run_vits_finetuning.py training_config_arabic.json
```

---

## 📌 Résumé

- Convertir le modèle MMS original avec `convert_original_discriminator_checkpoint.py`
- Créer un fichier de configuration JSON avec les bons attributs liés au dataset
- Si vous chargez le dataset localement, mettre à jour `dataset_name` avec le chemin local
- Vérifier que le vocabulaire du dataset correspond au vocabulaire du modèle (fichier `vocab.json`)
- Lancer le fine-tuning avec `accelerate`

---
