# Reconnaissance Automatique de la Parole (ASR) et Analyse de Sentiment

## Table des Matières
- [Introduction](#introduction)
- [Partie 1: Reconnaissance Automatique de la Parole (ASR)](#partie-1-reconnaissance-automatique-de-la-parole-asr)
  - [Installation des dépendances](#installation-des-dépendances)
  - [Prétraitement de l'audio](#prétraitement-de-laudio)
  - [Transcription de l'audio](#transcription-de-laudio)
- [Partie 2: Analyse de Sentiment](#partie-2-analyse-de-sentiment)
  - [Chargement des données et Tokenisation](#chargement-des-données-et-tokenisation)
  - [Entraînement du modèle BERT](#entrainement-du-modèle-bert)
  - [Évaluation du modèle](#évaluation-du-modèle)
  - [Analyse de Sentiment sur la Transcription](#analyse-de-sentiment-sur-la-transcription)
- [Conclusion](#conclusion)

## Introduction

Ce projet se compose de deux parties distinctes :

1. **Reconnaissance Automatique de la Parole (ASR)** : Utilisation d'un modèle Wav2Vec2 pour transcrire un fichier audio en texte.
2. **Analyse de Sentiment** : Utilisation d'un modèle distlBERT pour analyser le sentiment de la transcription générée.

Ce rapport documente en détail chaque étape du projet, en fournissant du code bien commenté pour faciliter la compréhension et la reproduction des résultats. 
L’Architecture complète de l’inférence est la suivante:
Audio --> ASR modèle --> Transcription --> sentimental analysis --> { transcription : ‘’bonjour , je suis content davoir étudié à DIT’’, 
                                                                        sentiment : ‘’positive’’ }


Il y a lieu de rappeler que nous avons d'abord effectuer autres modèles que nous avons mis dans le dossier 'others' à savoir:
 "moussaKam/barthez-sentiment-classification" et "camembert-case". Hormis, au niveau de "camembert-case" où le sentiment est négatif, les modèles ont permis d'avoir un sentiment positif mais nous pouvons toujours améliorer la performance du modèle. 
 
 Pour arriver à ces résultats qui peuvent être encore améliorés nous avons testé plusieurs architecture de modèle ainsi que plusieurs valeurs pour les hyperparamètres.

---

## Partie 1: Reconnaissance Automatique de la Parole (ASR)

### Installation des dépendances

Avant de commencer, assurez-vous d'avoir installé les dépendances nécessaires en exécutant la commande suivante :

```bash
pip install transformers torch librosa
```

### Prétraitement de l'audio
Le prétraitement de l'audio consiste à charger le fichier audio et à le rééchantillonner à 16 kHz pour qu'il soit compatible avec le modèle Wav2Vec2.

```bash
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch

model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-french"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Utilisation de l'appareil : {device}")

# Charger le modèle et le processeur
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)

def load_and_preprocess_audio(file_path):
    """
    Charge et prétraite un fichier audio en le rééchantillonnant à 16 kHz.

    Args:
        file_path (str): Chemin vers le fichier audio à charger.

    Returns:
        numpy.ndarray: Tableau numpy contenant les données audio échantillonnées à 16 kHz.
    """
    # Charger l'audio à 16 kHz
    speech_array, _ = librosa.load(file_path, sr=16_000)
    return speech_array
```

### Transcription de l'audio
La transcription consiste à utiliser le modèle Wav2Vec2 pour convertir l'audio en texte.

```bash
def predict_transcription(speech_array):
    """
    Prédire la transcription à partir d'une séquence audio.

    Args:
        speech_array (numpy.ndarray): Tableau numpy contenant les données audio prétraitées.

    Returns:
        str: La transcription textuelle prédite pour l'audio fourni.
    """
    # Préparer les inputs pour le modèle
    inputs = processor(speech_array, sampling_rate=16_000, return_tensors="pt", padding=True)

    # Transférer les inputs sur le bon device (GPU ou CPU)
    input_values = inputs.input_values.to(device)
    attention_mask = inputs.attention_mask.to(device)

    # Effectuer la prédiction sans gradient (inférence)
    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits

    # Décoder les prédictions pour obtenir le texte transcrit
    pred_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(pred_ids)[0]
    
    return transcription

# Exécuter la transcription
file_path = "C:/Users/USER/Documents/Audacity/dit.wav"  
speech_array = load_and_preprocess_audio(file_path)
transcription = predict_transcription(speech_array)
print("Transcription:", transcription)
```


## Partie 2: Analyse de Sentiment
### Chargement des données et Tokenisation
La première étape de l'analyse de sentiment consiste à charger les données de critiques de films et à les tokeniser pour les rendre compatibles avec le modèle BERT. Nous avons réduit proportionnellement les données initiales du Dataset https://www.kaggle.com/datasets/djilax/allocine-frenchmovie-reviews et retenu un pourcentage de 0.002 soit 320 lignes pour le train et 40 pour le test.

```bash
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification

class AllocineReviewsDataset(Dataset):
    def __init__(self, csv_file, device, model_name_or_path="distilbert/distilbert-base-multilingual-cased", max_length=128):
        self.device = device
        self.df = pd.read_csv(csv_file)
        self.labels = self.df.polarity.unique()
        labels_dict = {l: indx for indx, l in enumerate(self.labels)}
        
        self.df["polarity"] = self.df["polarity"].map(labels_dict)
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained(model_name_or_path)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        review_text = self.df.review[index]
        label_review = self.df.polarity[index]

        inputs = self.tokenizer(
            review_text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        labels = torch.tensor(label_review)

        return {
            "input_ids": inputs["input_ids"].squeeze(0).to(self.device),
            "attention_mask": inputs["attention_mask"].squeeze(0).to(self.device),
            "labels": labels.to(self.device)
        }

```

### Entraînement du modèle BERT
Le modèle BERT est ensuite entraîné sur le dataset de critiques de films.

```bash
class SentimentAnalysisBert(nn.Module):
    def __init__(self, model_name_or_path="distilbert/distilbert-base-multilingual-cased", n_classes=2):
        super(SentimentAnalysisBert, self).__init__()
        self.bert_pretained = BertForSequenceClassification.from_pretrained(model_name_or_path, num_labels=n_classes)

    def forward(self, input_ids, attention_mask):
        x = self.bert_pretained(input_ids=input_ids, attention_mask=attention_mask).logits
        return x

def training_step(model, train_loader, loss_fn, optimizer):
    """
    Entraîne le modèle sur les données d'entraînement.

    Args:
        model (nn.Module): Le modèle BERT à entraîner.
        train_loader (DataLoader): Le DataLoader contenant les données d'entraînement.
        loss_fn (nn.Module): La fonction de perte utilisée pour l'entraînement.
        optimizer (torch.optim.Optimizer): L'optimiseur utilisé pour l'entraînement.

    Returns:
        float: La perte moyenne sur l'ensemble du dataset d'entraînement.
    """
    model.train()
    total_loss = 0

    for data in tqdm(train_loader, total=len(train_loader)):
        input_ids = data["input_ids"]
        attention_mask = data["attention_mask"]
        labels = data["labels"]

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    return total_loss / len(train_loader.dataset)

```

### Évaluation du modèle
Le modèle est évalué sur un ensemble de test pour mesurer sa performance.

```bash
def evaluation_step(model, test_loader, loss_fn):
    """
    Évalue le modèle sur les données de test.

    Args:
        model (nn.Module): Le modèle BERT à évaluer.
        test_loader (DataLoader): Le DataLoader contenant les données de test.
        loss_fn (nn.Module): La fonction de perte utilisée pour l'évaluation.

    Returns:
        tuple: La perte moyenne, l'exactitude, les prédictions et les labels réels.
    """
    model.eval()
    correct_predictions = 0
    losses = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in tqdm(test_loader, total=len(test_loader)):
            input_ids = data["input_ids"]
            attention_mask = data["attention_mask"]
            labels = data["labels"]

            logits = model(input_ids, attention_mask=attention_mask)

            # Appliquer softmax pour obtenir les probabilités
            pred = torch.softmax(logits, dim=-1)
            
            # Obtenir les indices des classes les plus probables
            pred_class = torch.argmax(pred, dim=1)

            all_preds.extend(pred_class.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            loss = loss_fn(logits, labels)
            losses.append(loss.item())

            correct_predictions += torch.sum(pred_class == labels).item()

    accuracy = correct_predictions / len(test_loader.dataset)
    avg_loss = np.mean(losses)

    return avg_loss, accuracy, all_preds, all_labels

```

### Analyse de Sentiment sur la Transcription
Enfin, une fonction a été définie afin que la transcription obtenue dans la première partie soit analysée pour déterminer le sentiment.

```bash
def sentiment_analysis_on_transcription(model, tokenizer, transcription):
    """
    Analyse le sentiment d'une transcription donnée en utilisant un modèle BERT pré-entraîné.

    Args:
        model (nn.Module): Le modèle BERT pour l'analyse de sentiment.
        tokenizer (BertTokenizer): Le tokenizer BERT correspondant.
        transcription (str): Le texte transcrit à analyser.

    Returns:
        str: Le sentiment prédit (positif ou négatif).
    """
    inputs = tokenizer(transcription, return_tensors="pt", padding=True, truncation=True, max_length=128)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    
    # Calculer les probabilités des classes et choisir la plus haute
    sentiment_id = torch.argmax(torch.softmax(logits, dim=1)).item()
    sentiment = "Positif" if sentiment_id == 1 else "Négatif"

    return sentiment

# Exécuter l'analyse de sentiment
sentiment = sentiment_analysis_on_transcription(model, tokenizer, transcription)
print("Sentiment de la transcription:", sentiment)

```

## Conclusion

Ce projet a démontré comment combiner la Reconnaissance Automatique de la Parole (ASR) avec l'Analyse de Sentiment en utilisant des modèles pré-entraînés de Wav2Vec2 et BERT. Les sections de code fournies, avec des commentaires détaillés, permettent de comprendre chaque étape du processus, du prétraitement de l'audio à l'analyse du sentiment du texte transcrit.
Par ailleurs, une extension du projet a été entrepris qui consiste à faire un push du modèle sur huggingface_hub et l'utiliser afin 
```bash
```


```bash
```

```bash
```

```bash
```