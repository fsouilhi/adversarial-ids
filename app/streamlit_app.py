"""
Démonstration interactive — Attaques Adversariales sur les IDS
Jeu de données : NSL-KDD | Modèle : MLP PyTorch
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration de la page
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Adversarial IDS — NSL-KDD",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background: #1e293b;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        border-left: 3px solid #60a5fa;
    }
    .metric-card h3 { margin: 0; font-size: 0.8rem; color: #94a3b8; letter-spacing: 0.05em; }
    .metric-card p  { margin: 0.2rem 0 0; font-size: 1.6rem; font-weight: 700; color: #f1f5f9; }
    .attack-card {
        background: #1e293b;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        border-left: 3px solid #f87171;
    }
    .defense-card {
        background: #1e293b;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        border-left: 3px solid #34d399;
    }
    section[data-testid="stSidebar"] { background: #0f172a; }
    section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    section[data-testid="stSidebar"] label { color: #f1f5f9 !important; font-weight: 500; }
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { color: #f1f5f9 !important; font-weight: 700; }
    section[data-testid="stSidebar"] [data-baseweb="select"] { background: #1e293b !important; }
    section[data-testid="stSidebar"] [data-baseweb="select"] * { color: #1e293b !important; }
    section[data-testid="stSidebar"] [data-baseweb="select"] [data-testid="stMarkdownContainer"] * { color: #1e293b !important; }
    section[data-testid="stSidebar"] div[role="listbox"] * { color: #1e293b !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

NOMS_COLONNES = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count",
    "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty_level",
]

FEATURES_CAT = ["protocol_type", "service", "flag"]

URL_TRAIN = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt"
URL_TEST  = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt"

DEVICE = torch.device("cpu")
GRAINE = 42
torch.manual_seed(GRAINE)
np.random.seed(GRAINE)

# ---------------------------------------------------------------------------
# Modèle
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, dim_entree: int, taux_dropout: float = 0.3):
        super().__init__()
        self.reseau = nn.Sequential(
            nn.Linear(dim_entree, 128), nn.ReLU(), nn.Dropout(taux_dropout),
            nn.Linear(128, 64),         nn.ReLU(), nn.Dropout(taux_dropout),
            nn.Linear(64, 32),          nn.ReLU(),
            nn.Linear(32, 1),           nn.Sigmoid(),
        )

    def forward(self, x):
        return self.reseau(x).squeeze(1)

# ---------------------------------------------------------------------------
# Attaques
# ---------------------------------------------------------------------------

def fgsm(modele, X, y, epsilon, critere):
    X_adv = X.clone().detach().requires_grad_(True)
    perte = critere(modele(X_adv), y)
    modele.zero_grad()
    perte.backward()
    X_adv = (X_adv.detach() + epsilon * X_adv.grad.sign()).clamp(0, 1)
    return X_adv.detach()


def pgd(modele, X, y, epsilon, alpha, nb_iterations, critere):
    X_adv = (X.clone().detach() + torch.empty_like(X).uniform_(-epsilon, epsilon)).clamp(0, 1)
    for _ in range(nb_iterations):
        X_adv.requires_grad_(True)
        perte = critere(modele(X_adv), y)
        modele.zero_grad()
        perte.backward()
        with torch.no_grad():
            delta = (X_adv + alpha * X_adv.grad.sign() - X).clamp(-epsilon, epsilon)
            X_adv = (X + delta).clamp(0, 1).detach()
    return X_adv


# ---------------------------------------------------------------------------
# Défenses
# ---------------------------------------------------------------------------

def arrondi(X, nb_decimales):
    f = 10 ** nb_decimales
    return torch.round(X * f) / f


def lissage_mediane(X, taille_fenetre):
    X_np = X.numpy()
    demi  = taille_fenetre // 2
    X_l   = np.copy(X_np)
    for j in range(X_np.shape[1]):
        debut, fin = max(0, j - demi), min(X_np.shape[1], j + demi + 1)
        X_l[:, j]  = np.median(X_np[:, debut:fin], axis=1)
    return torch.tensor(X_l, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Chargement des données et entraînement (mis en cache)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Chargement du jeu de données NSL-KDD…")
def charger_et_entrainer():
    df_train = pd.read_csv(URL_TRAIN, names=NOMS_COLONNES).drop(columns=["difficulty_level"])
    df_test  = pd.read_csv(URL_TEST,  names=NOMS_COLONNES).drop(columns=["difficulty_level"])

    df_train_enc = pd.get_dummies(df_train, columns=FEATURES_CAT)
    df_test_enc  = pd.get_dummies(df_test,  columns=FEATURES_CAT)

    colonnes = [c for c in df_train_enc.columns if c not in ("label",)]
    df_test_enc = df_test_enc.reindex(columns=colonnes, fill_value=0)

    y_train = (df_train["label"] != "normal").astype(int).values
    y_test  = (df_test["label"]  != "normal").astype(int).values

    feats = [c for c in colonnes if c != "label"]
    X_train_raw = df_train_enc[feats].values.astype(np.float32)
    X_test_raw  = df_test_enc[feats].values.astype(np.float32)

    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train_raw)
    X_test_s  = scaler.transform(X_test_raw)

    X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
    y_train_t = torch.tensor(y_train,   dtype=torch.float32)
    X_test_t  = torch.tensor(X_test_s,  dtype=torch.float32)
    y_test_t  = torch.tensor(y_test,    dtype=torch.float32)

    dim = X_train_t.shape[1]

    # --- Entraînement modèle standard ---
    modele = MLP(dim).to(DEVICE)
    critere = nn.BCELoss()
    opt = torch.optim.Adam(modele.parameters(), lr=1e-3)
    from torch.utils.data import DataLoader, TensorDataset
    loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=512, shuffle=True)

    meilleure, patience, compteur, meilleur_etat = float("inf"), 10, 0, None
    for _ in range(60):
        modele.train()
        for xb, yb in loader:
            opt.zero_grad()
            critere(modele(xb), yb).backward()
            opt.step()
        modele.eval()
        with torch.no_grad():
            pv = critere(modele(X_test_t), y_test_t).item()
        if pv < meilleure:
            meilleure, compteur = pv, 0
            meilleur_etat = {k: v.clone() for k, v in modele.state_dict().items()}
        else:
            compteur += 1
            if compteur >= patience:
                break
    modele.load_state_dict(meilleur_etat)

    # --- Entraînement modèle robuste ---
    modele_r = MLP(dim).to(DEVICE)
    opt_r = torch.optim.Adam(modele_r.parameters(), lr=1e-3)
    meilleure_r, compteur_r, meilleur_etat_r = float("inf"), 0, None
    for _ in range(60):
        modele_r.train()
        for xb, yb in loader:
            taille_adv = len(xb) // 2
            xb_adv = fgsm(modele_r, xb[:taille_adv], yb[:taille_adv], 0.1, critere)
            xb_mix = torch.cat([xb_adv, xb[taille_adv:]])
            opt_r.zero_grad()
            critere(modele_r(xb_mix), yb).backward()
            opt_r.step()
        modele_r.eval()
        with torch.no_grad():
            pv_r = critere(modele_r(X_test_t), y_test_t).item()
        if pv_r < meilleure_r:
            meilleure_r, compteur_r = pv_r, 0
            meilleur_etat_r = {k: v.clone() for k, v in modele_r.state_dict().items()}
        else:
            compteur_r += 1
            if compteur_r >= patience:
                break
    modele_r.load_state_dict(meilleur_etat_r)

    return modele, modele_r, X_test_t, y_test_t, critere


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

st.title("Attaques Adversariales sur les Systèmes de Détection d'Intrusions")
st.caption("NSL-KDD · MLP PyTorch · FGSM / PGD · Entraînement adversarial · Feature Squeezing")
st.divider()

modele, modele_r, X_test_t, y_test_t, critere = charger_et_entrainer()

# Sidebar
with st.sidebar:
    st.header("Paramètres")

    st.subheader("Attaque")
    type_attaque = st.selectbox("Type d'attaque", ["FGSM", "PGD"])
    epsilon = st.slider("Epsilon (ε)", min_value=0.0, max_value=0.5, value=0.1, step=0.01)
    if type_attaque == "PGD":
        nb_iter = st.slider("Nombre d'itérations", min_value=5, max_value=50, value=40, step=5)
        alpha   = st.slider("Pas α", min_value=0.001, max_value=0.05, value=0.01, step=0.001)

    st.subheader("Défense")
    type_defense = st.selectbox(
        "Mécanisme de défense",
        ["Aucune", "Entraînement adversarial", "Arrondi 1 décimale",
         "Arrondi 2 décimales", "Médiane fenêtre=3", "Médiane fenêtre=5"]
    )

    st.divider()
    nb_echantillons = st.slider(
        "Nombre d'échantillons de test", 500, len(X_test_t), 2000, step=500
    )
    lancer = st.button("Lancer l'évaluation", type="primary", use_container_width=True)

# Sélection d'un sous-ensemble
idx = torch.randperm(len(X_test_t))[:nb_echantillons]
X_sub = X_test_t[idx]
y_sub = y_test_t[idx]

# Évaluation de référence (données propres)
modele.eval()
with torch.no_grad():
    preds_propres = (modele(X_sub).numpy() >= 0.5).astype(int)
y_np = y_sub.numpy().astype(int)
acc_ref = accuracy_score(y_np, preds_propres)
f1_ref  = f1_score(y_np, preds_propres, zero_division=0)

# Affichage référence
st.subheader("Performances de référence (données propres)")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f'<div class="metric-card"><h3>PRÉCISION</h3><p>{acc_ref:.4f}</p></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-card"><h3>F1-SCORE</h3><p>{f1_ref:.4f}</p></div>', unsafe_allow_html=True)
with c3:
    nb_att = int(y_np.sum())
    st.markdown(f'<div class="metric-card"><h3>ÉCHANTILLONS</h3><p>{nb_echantillons} ({nb_att} attaques)</p></div>', unsafe_allow_html=True)

st.divider()

if lancer:
    with st.spinner("Génération des exemples adversariaux…"):
        # Génération de l'attaque
        modele.eval()
        if type_attaque == "FGSM":
            X_adv = fgsm(modele, X_sub, y_sub, epsilon, critere)
        else:
            X_adv = pgd(modele, X_sub, y_sub, epsilon, alpha, nb_iter, critere)

        # Application de la défense
        if type_defense == "Entraînement adversarial":
            modele_eval = modele_r
            X_eval = X_adv
        elif type_defense == "Arrondi 1 décimale":
            modele_eval = modele
            X_eval = arrondi(X_adv, 1)
        elif type_defense == "Arrondi 2 décimales":
            modele_eval = modele
            X_eval = arrondi(X_adv, 2)
        elif type_defense == "Médiane fenêtre=3":
            modele_eval = modele
            X_eval = lissage_mediane(X_adv, 3)
        elif type_defense == "Médiane fenêtre=5":
            modele_eval = modele
            X_eval = lissage_mediane(X_adv, 5)
        else:
            modele_eval = modele
            X_eval = X_adv

        modele_eval.eval()
        with torch.no_grad():
            preds_adv = (modele_eval(X_eval.to(DEVICE)).cpu().numpy() >= 0.5).astype(int)

        acc_adv = accuracy_score(y_np, preds_adv)
        f1_adv  = f1_score(y_np, preds_adv, zero_division=0)
        delta_acc = acc_adv - acc_ref
        delta_f1  = f1_adv  - f1_ref

    # Résultats après attaque/défense
    label_section = f"Après attaque {type_attaque} (ε={epsilon})"
    if type_defense != "Aucune":
        label_section += f" + Défense : {type_defense}"
    st.subheader(label_section)

    d1, d2, d3, d4 = st.columns(4)
    couleur_acc = "#f87171" if delta_acc < 0 else "#34d399"
    couleur_f1  = "#f87171" if delta_f1  < 0 else "#34d399"
    with d1:
        st.markdown(f'<div class="attack-card"><h3>PRÉCISION</h3><p>{acc_adv:.4f}</p></div>', unsafe_allow_html=True)
    with d2:
        st.markdown(f'<div class="attack-card"><h3>F1-SCORE</h3><p>{f1_adv:.4f}</p></div>', unsafe_allow_html=True)
    with d3:
        signe_acc = "+" if delta_acc >= 0 else ""
        st.markdown(f'<div class="metric-card"><h3>DELTA PRÉCISION</h3><p style="color:{couleur_acc}">{signe_acc}{delta_acc:.4f}</p></div>', unsafe_allow_html=True)
    with d4:
        signe_f1 = "+" if delta_f1 >= 0 else ""
        st.markdown(f'<div class="metric-card"><h3>DELTA F1-SCORE</h3><p style="color:{couleur_f1}">{signe_f1}{delta_f1:.4f}</p></div>', unsafe_allow_html=True)

    st.divider()

    # Graphiques
    col_left, col_right = st.columns(2)

    # Comparaison des métriques
    with col_left:
        st.markdown("**Comparaison des métriques**")
        fig, ax = plt.subplots(figsize=(5, 3.5), facecolor="#0f172a")
        ax.set_facecolor("#0f172a")
        x     = np.arange(2)
        width = 0.3
        labels_metriques = ["Précision", "F1-score"]
        vals_ref = [acc_ref, f1_ref]
        vals_adv = [acc_adv, f1_adv]
        b1 = ax.bar(x - width / 2, vals_ref, width, label="Référence", color="#60a5fa")
        b2 = ax.bar(x + width / 2, vals_adv, width, label=f"{type_attaque}" + (f" + {type_defense}" if type_defense != "Aucune" else ""), color="#f87171" if delta_acc < 0 else "#34d399")
        ax.set_xticks(x)
        ax.set_xticklabels(labels_metriques, color="#e2e8f0")
        ax.set_ylim(0, 1.1)
        ax.tick_params(colors="#94a3b8")
        ax.spines[:].set_color("#334155")
        for b in list(b1) + list(b2):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                    f"{b.get_height():.3f}", ha="center", fontsize=8, color="#e2e8f0")
        ax.legend(fontsize=8, facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Matrices de confusion côte à côte
    with col_right:
        st.markdown("**Matrice de confusion — après attaque**")
        fig, ax = plt.subplots(figsize=(4, 3.5), facecolor="#0f172a")
        ax.set_facecolor("#0f172a")
        mat = confusion_matrix(y_np, preds_adv)
        im  = ax.imshow(mat, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Normal", "Attaque"], color="#e2e8f0")
        ax.set_yticklabels(["Normal", "Attaque"], color="#e2e8f0")
        ax.set_xlabel("Prédit", color="#94a3b8")
        ax.set_ylabel("Réel", color="#94a3b8")
        ax.spines[:].set_color("#334155")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(mat[i, j]), ha="center", va="center",
                        color="white", fontsize=12, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Courbe de dégradation selon epsilon
    st.divider()
    st.markdown("**Dégradation de la précision en fonction de ε**")

    epsilons = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    accs_std, accs_def = [], []

    with st.spinner("Calcul de la courbe de dégradation…"):
        for eps in epsilons:
            if eps == 0.0:
                accs_std.append(acc_ref)
                accs_def.append(acc_ref)
                continue
            if type_attaque == "FGSM":
                X_a = fgsm(modele, X_sub, y_sub, eps, critere)
            else:
                X_a = pgd(modele, X_sub, y_sub, eps, 0.01, 20, critere)

            modele.eval()
            with torch.no_grad():
                p_std = (modele(X_a).numpy() >= 0.5).astype(int)
            accs_std.append(accuracy_score(y_np, p_std))

            if type_defense == "Entraînement adversarial":
                modele_r.eval()
                with torch.no_grad():
                    p_def = (modele_r(X_a).numpy() >= 0.5).astype(int)
            elif type_defense == "Arrondi 1 décimale":
                modele.eval()
                with torch.no_grad():
                    p_def = (modele(arrondi(X_a, 1)).numpy() >= 0.5).astype(int)
            elif type_defense == "Arrondi 2 décimales":
                modele.eval()
                with torch.no_grad():
                    p_def = (modele(arrondi(X_a, 2)).numpy() >= 0.5).astype(int)
            elif type_defense == "Médiane fenêtre=3":
                modele.eval()
                with torch.no_grad():
                    p_def = (modele(lissage_mediane(X_a, 3)).numpy() >= 0.5).astype(int)
            elif type_defense == "Médiane fenêtre=5":
                modele.eval()
                with torch.no_grad():
                    p_def = (modele(lissage_mediane(X_a, 5)).numpy() >= 0.5).astype(int)
            else:
                p_def = p_std
            accs_def.append(accuracy_score(y_np, p_def))

    fig, ax = plt.subplots(figsize=(10, 3.5), facecolor="#0f172a")
    ax.set_facecolor("#0f172a")
    ax.plot(epsilons, accs_std, marker="o", color="#f87171", label="Sans défense", linewidth=2)
    if type_defense != "Aucune":
        ax.plot(epsilons, accs_def, marker="s", color="#34d399", label=type_defense, linewidth=2)
    ax.axvline(x=epsilon, color="#fbbf24", linestyle="--", alpha=0.6, label=f"ε sélectionné ({epsilon})")
    ax.set_xlabel("Epsilon (ε)", color="#94a3b8")
    ax.set_ylabel("Précision", color="#94a3b8")
    ax.set_ylim(0, 1.05)
    ax.tick_params(colors="#94a3b8")
    ax.spines[:].set_color("#334155")
    ax.legend(fontsize=9, facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

else:
    st.info("Configure les paramètres dans la barre latérale et clique sur **Lancer l'évaluation**.")
