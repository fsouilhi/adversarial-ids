# Attaques Adversariales sur les Systèmes de Détection d'Intrusions

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-yellow)
![License](https://img.shields.io/badge/Licence-MIT-green)

## Présentation

Ce projet étudie la vulnérabilité des systèmes de détection d'intrusions (IDS) basés sur l'apprentissage automatique face aux perturbations adversariales, et évalue deux stratégies de défense.

Un perceptron multi-couches (MLP) est entraîné sur le jeu de données **NSL-KDD** pour classifier le trafic réseau en normal ou malveillant. Deux attaques en boîte blanche sont ensuite appliquées — **FGSM** et **PGD** — puis contrecarrées par un **entraînement adversarial** et du **feature squeezing**.

---

## Démo interactive

> **[Lancer la démo Streamlit](https://adversarial-ids-gya5wtwnntqdkelbnrgvch.streamlit.app)** ← lien de déploiement à ajouter

La démo permet de visualiser en temps réel l'effet des perturbations adversariales sur les prédictions du modèle et de comparer les défenses.

---

## Structure du dépôt

```
adversarial-ids/
├── notebook/
│   └── adversarial_ids.ipynb      # Notebook principal (Google Colab)
├── app/
│   └── streamlit_app.py           # Application de démonstration
├── data/                          # Figures générées (gitignorées)
├── requirements.txt
└── README.md
```

---

## Méthodologie

### 1. Jeu de données — NSL-KDD

Le jeu de données NSL-KDD est une version améliorée du KDD Cup 1999, largement utilisé comme référence pour l'évaluation des IDS. Il contient 41 features de trafic réseau et 5 classes : trafic normal et quatre catégories d'attaques (DoS, Probe, R2L, U2R).

Pour ce projet, le problème est formulé en **classification binaire** : normal vs. attaque.

### 2. Modèle MLP

Architecture retenue :

| Couche | Neurones | Activation | Régularisation |
|--------|----------|------------|----------------|
| Entrée | `dim_entrée` | — | — |
| Cachée 1 | 128 | ReLU | Dropout 0.3 |
| Cachée 2 | 64 | ReLU | Dropout 0.3 |
| Cachée 3 | 32 | ReLU | — |
| Sortie | 1 | Sigmoid | — |

Optimiseur : Adam (`lr=1e-3`) — Perte : Binary Cross-Entropy — Arrêt anticipé : patience 10

### 3. Attaques adversariales

**FGSM** (*Fast Gradient Sign Method*, Goodfellow et al., 2014) :

$$x_{adv} = x + \varepsilon \cdot \text{sign}\left(\nabla_x \mathcal{L}(f_\theta(x), y)\right)$$

**PGD** (*Projected Gradient Descent*, Madry et al., 2018) — version itérative de FGSM avec projection dans la boule $\ell_\infty(\varepsilon)$ et initialisation aléatoire.

### 4. Défenses

- **Entraînement adversarial** : réentraînement sur un mélange d'exemples propres et d'exemples FGSM générés à la volée (50/50 par batch).
- **Feature Squeezing** : arrondi à précision réduite et lissage par médiane, sans réentraînement du modèle.

---

## Résultats

| Modèle | Données | Précision | F1-score |
|--------|---------|-----------|----------|
| MLP standard | Propres | 0.7977 | 0.7897 |
| MLP standard | FGSM (ε=0.1) | 0.2851 | 0.4437 |
| MLP standard | PGD (ε=0.1) | 0.3386 | 0.5059 |
| Entraînement adversarial | Propres | 0.7857 | 0.7827 |
| Entraînement adversarial | FGSM (ε=0.1) | 0.7444 | 0.7372 |
| Entraînement adversarial | PGD (ε=0.1) | 0.7506 | 0.7459 |
| Feature Squeezing — Arrondi 1 décimale | Propres | 0.7886 | 0.7785 |
| Feature Squeezing — Arrondi 1 décimale | FGSM (ε=0.1) | 0.2848 | 0.4431 |
| Feature Squeezing — Arrondi 1 décimale | PGD (ε=0.1) | 0.3744 | 0.5197 |
| Feature Squeezing — Arrondi 2 décimales | Propres | 0.7996 | 0.7921 |
| Feature Squeezing — Arrondi 2 décimales | FGSM (ε=0.1) | 0.2853 | 0.4438 |
| Feature Squeezing — Arrondi 2 décimales | PGD (ε=0.1) | 0.3386 | 0.5059 |
| Feature Squeezing — Médiane fenêtre=3 | Propres | 0.7434 | 0.8043 |
| Feature Squeezing — Médiane fenêtre=3 | FGSM (ε=0.1) | 0.3415 | 0.5090 |
| Feature Squeezing — Médiane fenêtre=3 | PGD (ε=0.1) | 0.4309 | 0.6023 |
| Feature Squeezing — Médiane fenêtre=5 | Propres | 0.4318 | 0.4158 |
| Feature Squeezing — Médiane fenêtre=5 | FGSM (ε=0.1) | 0.1463 | 0.2521 |
| Feature Squeezing — Médiane fenêtre=5 | PGD (ε=0.1) | 0.4333 | 0.6034 |

---

## Reproduire les résultats

### Sur Google Colab (recommandé)

1. Ouvrir le notebook : **[Ouvrir dans Colab](#)**
2. Exécuter toutes les cellules : `Exécution → Tout exécuter`

Aucune installation manuelle requise — toutes les dépendances sont gérées par la première cellule du notebook.

### En local

```bash
git clone https://github.com/fsouilhi/adversarial-ids.git
cd adversarial-ids
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebook/adversarial_ids.ipynb
```

---

## Dépendances

```
torch>=2.0
scikit-learn>=1.3
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
streamlit>=1.30
```

---

## Références

- Goodfellow, I. et al. (2014). *Explaining and Harnessing Adversarial Examples*. ICLR 2015.
- Madry, A. et al. (2018). *Towards Deep Learning Models Resistant to Adversarial Attacks*. ICLR 2018.
- Xu, W. et al. (2018). *Feature Squeezing: Detecting Adversarial Examples in Deep Neural Networks*. NDSS 2018.
- Tavallaee, M. et al. (2009). *A Detailed Analysis of the KDD CUP 99 Data Set*. IEEE CISDA 2009.

---

## Auteur

**Fatima Souilhi** — [@fsouilhi](https://github.com/fsouilhi)  

