import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ----------------------
# Configuration
# ----------------------
DRUGS_CSV = os.path.join(PROJECT_ROOT, 'data', 'processed', 'drug_properties_realistic.csv')
MEDS_CSV = os.path.join(PROJECT_ROOT, 'data', 'processed', 'enhanced_medicine_data.csv')
N_CLUSTERS_DRUGS = 15
N_CLUSTERS_MEDS_USES = 15
N_CLUSTERS_MEDS_SIDE_EFFECTS = 15
RANDOM_STATE = 42

# ----------------------
# Utilities
# ----------------------
def ensure_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

# ----------------------
# Drug clustering & plotting
# ----------------------
def cluster_and_plot_drugs(df):
    if 'drug' in df.columns:
        df = df.set_index('drug')

    features = ['molecular_weight', 'h_donors', 'h_acceptors', 'rotatable_bonds', 'logp']
    data = df[features].dropna()
    print(f"Clustering {len(data)} drugs into {N_CLUSTERS_DRUGS} colonies.")

    km = KMeans(n_clusters=N_CLUSTERS_DRUGS, random_state=RANDOM_STATE)
    data['colony'] = km.fit_predict(data)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(data[features])

    plt.figure(figsize=(8, 6))
    for c in range(N_CLUSTERS_DRUGS):
        mask = data['colony'] == c
        plt.scatter(coords[mask, 0], coords[mask, 1], label=f'Colony {c}', alpha=0.7)
    plt.title('Drug Colonies by Physicochemical Properties')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.tight_layout()
    plt.savefig('drug_colony_scatter.png')
    plt.show()

    counts = data['colony'].value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    counts.plot(kind='bar')
    plt.title('Number of Drugs per Colony')
    plt.xlabel('Colony')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('drug_colony_counts.png')
    plt.show()

    return data

# ----------------------
# Medicine clustering & plotting (uses)
# ----------------------
def cluster_and_plot_medicines_by_uses(df):
    uses = df['uses'].fillna('')
    print(f"Clustering {len(uses)} medicines by 'uses' into {N_CLUSTERS_MEDS_USES} groups.")

    vec = TfidfVectorizer(max_features=500, stop_words='english')
    X = vec.fit_transform(uses)

    km = KMeans(n_clusters=N_CLUSTERS_MEDS_USES, random_state=RANDOM_STATE)
    df['uses_group'] = km.fit_predict(X)

    # Bar chart
    counts = df['uses_group'].value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    counts.plot(kind='bar')
    plt.title('Number of Medicines per Uses-Cluster')
    plt.xlabel('Uses-Cluster')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('medicine_uses_group_counts.png')
    plt.show()

    # Scatter plot
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X.toarray())
    plt.figure(figsize=(8, 6))
    for c in range(N_CLUSTERS_MEDS_USES):
        mask = df['uses_group'] == c
        plt.scatter(coords[mask, 0], coords[mask, 1], label=f'Group {c}', alpha=0.7)
    plt.title('Medicine Uses Clusters (TF-IDF + PCA)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.tight_layout()
    plt.savefig('medicine_uses_group_scatter.png')
    plt.show()

    return df

# ----------------------
# Medicine clustering & plotting (side effects)
# ----------------------
def cluster_and_plot_medicines_by_side_effects(df):
    sefs = df['side_effects'].fillna('')
    print(f"Clustering {len(sefs)} medicines by 'side_effects' into {N_CLUSTERS_MEDS_SIDE_EFFECTS} groups.")

    vec = TfidfVectorizer(max_features=500, stop_words='english')
    X = vec.fit_transform(sefs)

    km = KMeans(n_clusters=N_CLUSTERS_MEDS_SIDE_EFFECTS, random_state=RANDOM_STATE)
    df['side_effects_group'] = km.fit_predict(X)

    # Bar chart
    counts = df['side_effects_group'].value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    counts.plot(kind='bar')
    plt.title('Number of Medicines per Side-Effects-Cluster')
    plt.xlabel('Side-Effects-Cluster')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('medicine_side_effects_group_counts.png')
    plt.show()

    # Scatter plot
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X.toarray())
    plt.figure(figsize=(8, 6))
    for c in range(N_CLUSTERS_MEDS_SIDE_EFFECTS):
        mask = df['side_effects_group'] == c
        plt.scatter(coords[mask, 0], coords[mask, 1], label=f'Group {c}', alpha=0.7)
    plt.title('Medicine Side Effects Clusters (TF-IDF + PCA)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.tight_layout()
    plt.savefig('medicine_side_effects_group_scatter.png')
    plt.show()

    return df

# ----------------------
# Main
# ----------------------
def main():
    ensure_file(DRUGS_CSV)
    ensure_file(MEDS_CSV)

    drugs = pd.read_csv(DRUGS_CSV)
    meds  = pd.read_csv(MEDS_CSV)

    clustered_drugs = cluster_and_plot_drugs(drugs)
    meds = cluster_and_plot_medicines_by_uses(meds)
    meds = cluster_and_plot_medicines_by_side_effects(meds)

    print("All visualizations generated:")
    print(" - drug_colony_scatter.png")
    print(" - drug_colony_counts.png")
    print(" - medicine_uses_group_counts.png")
    print(" - medicine_uses_group_scatter.png")
    print(" - medicine_side_effects_group_counts.png")
    print(" - medicine_side_effects_group_scatter.png")

if __name__ == '__main__':
    main()
