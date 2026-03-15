# Attaques Adversariales sur les Systèmes de Détection d'Intrusions

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-yellow)
![License](https://img.shields.io/badge/Licence-MIT-green)

## Présentation

Ce projet étudie la vulnérabilité des systèmes de détection d'intrusions (IDS) basés sur l'apprentissage automatique face aux perturbations adversariales, et évalue deux stratégies de défense.

Un perceptron multi-couches (MLP) est entraîné sur le jeu de données **NSL-KDD** pour classifier le trafic réseau en trafic normal ou malveillant. Deux attaques en boîte blanche (*white-box*) sont ensuite appliquées — **FGSM** et **PGD** — puis contrecarrées par un **entraînement adversarial** et du **feature squeezing**.

---

## Démo interactive

> **[Lancer la démo Streamlit](https://adversarial-ids-gya5wtwnntqdkelbnrgvch.streamlit.app)**

La démo permet de visualiser en temps réel l'effet des perturbations adversariales sur les prédictions du modèle, de faire varier le budget de perturbation ε et de comparer les mécanismes de défense.

---

## Structure du dépôt

```
adversarial-ids/
├── notebook/
│   └── adversarial_ids.ipynb      # Notebook principal (Google Colab)
├── app/
│   └── streamlit_app.py           # Application de démonstration Streamlit
├── data/                          # Figures générées (gitignorées)
├── requirements.txt
└── README.md
```

---

## Méthodologie

### 1. Jeu de données — NSL-KDD

Le jeu de données NSL-KDD est une version améliorée du KDD Cup 1999, corrigeant les problèmes d'enregistrements redondants et de déséquilibre de classes qui affectaient le jeu original. Il constitue une référence standard pour l'évaluation des IDS dans la littérature académique.

**Caractéristiques générales :**

| | Entraînement | Test |
|---|---|---|
| Échantillons | 125 973 | 22 544 |
| Features | 41 | 41 |
| Classes | 5 | 5 |

**Types de features :**

- **Features de base** (9) : caractéristiques de la connexion TCP/IP individuelle — durée, protocole, service, octets transférés, flags, etc.
- **Features de contenu** (13) : informations extraites des données utiles (*payload*) — tentatives de connexion échouées, commandes root, accès aux fichiers, etc.
- **Features temporelles** (9) : statistiques calculées sur une fenêtre de 2 secondes — taux d'erreurs SYN, taux de services différents, etc.
- **Features d'hôte** (10) : statistiques sur les 100 dernières connexions vers le même hôte destination.

**Features catégorielles :**

| Feature | Modalités |
|---|---|
| `protocol_type` | tcp, udp, icmp |
| `service` | 70 services réseau (http, ftp, smtp, ssh, …) |
| `flag` | 11 états de connexion (SF, S0, REJ, RSTO, …) |

**Catégories d'attaques :**

| Catégorie | Description | Exemples |
|---|---|---|
| DoS | Déni de service | neptune, smurf, back, teardrop |
| Probe | Reconnaissance réseau | ipsweep, nmap, satan, portsweep |
| R2L | Accès distant non autorisé | guess_passwd, ftp_write, imap |
| U2R | Élévation de privilèges locale | buffer_overflow, rootkit, perl |

Pour ce projet, le problème est formulé en **classification binaire** : `normal` (0) vs. `attaque` (1).

**Observations EDA :**
- Le trafic DoS représente la catégorie d'attaque la plus fréquente (~45 % des attaques en entraînement).
- La classe normale et la classe attaque sont globalement équilibrées dans NSL-KDD, contrairement au KDD Cup 99 original.
- Plusieurs features de taux (*rate features*) présentent de fortes corrélations entre elles (ex. `serror_rate` / `srv_serror_rate`, `dst_host_serror_rate` / `dst_host_srv_serror_rate`).
- Les features `num_outbound_cmds` et `is_host_login` sont quasi-constantes (variance nulle) et n'apportent pas d'information discriminante.

---

### 2. Prétraitement

| Étape | Détail |
|---|---|
| Encodage one-hot | `protocol_type`, `service`, `flag` — alignement des colonnes train/test par `reindex` |
| Binarisation des étiquettes | `normal → 0`, toute attaque `→ 1` |
| Normalisation | Min-Max [0, 1] — scaler ajusté sur le train uniquement (pas de *data leakage*) |
| Découpage validation | 85 % / 15 %, stratifié, `random_state=42` |
| Conversion | Tenseurs PyTorch `float32` |

Après encodage one-hot, le vecteur de features atteint **~122 dimensions**.

---

### 3. Modèle MLP

Architecture retenue :

| Couche | Neurones | Activation | Régularisation |
|--------|----------|------------|----------------|
| Entrée | ~122 | — | — |
| Cachée 1 | 128 | ReLU | Dropout 0.3 |
| Cachée 2 | 64 | ReLU | Dropout 0.3 |
| Cachée 3 | 32 | ReLU | — |
| Sortie | 1 | Sigmoid | — |

**Entraînement :** optimiseur Adam (`lr=1e-3`), perte Binary Cross-Entropy, batch size 512, arrêt anticipé (patience=10 époques sur la perte de validation), restauration du meilleur état.

---

### 4. Attaques adversariales

Les attaques sont formulées en **boîte blanche** (*white-box*) : l'adversaire dispose d'un accès complet à l'architecture et aux paramètres du modèle. Les perturbations sont bornées en norme $\ell_\infty$ et les exemples adversariaux sont maintenus dans $[0, 1]$.

**FGSM** (*Fast Gradient Sign Method*, Goodfellow et al., 2014) — perturbation en un seul pas :

$$x_{adv} = x + \varepsilon \cdot \text{sign}\left(\nabla_x \mathcal{L}(f_\theta(x), y)\right)$$

**PGD** (*Projected Gradient Descent*, Madry et al., 2018) — version itérative avec initialisation aléatoire et projection dans la boule $\ell_\infty(\varepsilon)$ :

$$x^{(t+1)}_{adv} = \Pi_{\mathcal{B}_\varepsilon(x)} \left( x^{(t)}_{adv} + \alpha \cdot \text{sign}\left(\nabla_{x^{(t)}_{adv}} \mathcal{L}\right) \right)$$

---

### 5. Mécanismes de défense

**Entraînement adversarial** : le modèle est réentraîné sur un mélange 50/50 d'exemples propres et d'exemples FGSM générés à la volée (`ε=0.1`). Cette approche, introduite par Madry et al. (2018), est considérée comme la défense empirique la plus robuste.

**Feature Squeezing** (Xu et al., 2018) : les features sont lissées avant inférence afin de supprimer les perturbations à faible amplitude, sans réentraînement du modèle. Deux variantes sont testées :
- Arrondi à précision réduite (1 et 2 décimales)
- Lissage par médiane (fenêtre de taille 3 et 5)

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

**Analyse :** Le MLP standard subit une dégradation sévère sous attaque (précision 0.80 → 0.29 sous FGSM). L'entraînement adversarial maintient une précision stable (~0.74–0.75) face aux deux types d'attaques, au prix d'une légère réduction sur données propres (−0.012). Le feature squeezing apporte un gain marginal sous attaque mais reste insuffisant face à PGD. La médiane avec fenêtre=5 dégrade significativement les performances sur données propres, indiquant un lissage excessif.

---

## Reproduire les résultats

### Sur Google Colab (recommandé)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fsouilhi/adversarial-ids/blob/main/notebook/adversarial_ids.ipynb)

Exécuter toutes les cellules : `Exécution → Tout exécuter`. Toutes les dépendances sont gérées par la première cellule du notebook.

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
