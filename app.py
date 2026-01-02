import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# 1. INISIALISASI & KONFIGURASI
np.random.seed(42)

# Daftar nama fitur asli sesuai dokumentasi UCI Wisconsin Breast Cancer
feature_names = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

# 2. PREPROCESSING DATA 
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
# Menentukan nama kolom (ID dan Diagnosis diikuti 30 fitur)
columns = ['ID', 'Diagnosis'] + feature_names
df = pd.read_csv(url, header=None, names=columns)

# Encoding Target: Malignant (1), Benign (0)
df['Diagnosis'] = df['Diagnosis'].map({'M': 1, 'B': 0})

# Memisahkan Fitur dan Target (Menggunakan 'ID' dan 'Diagnosis' sesuai definisi columns)
X = df.drop(['ID', 'Diagnosis'], axis=1).values
y = df['Diagnosis'].values

# Normalisasi Data menggunakan StandardScaler 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Sistem Siap. Dataset: {X_scaled.shape[0]} sampel, {X_scaled.shape[1]} fitur.")


# 3. IMPLEMENTASI GENETIC ALGORITHM (GA) 
# Parameter sesuai Metodologi 
POP_SIZE = 50           # Ukuran Populasi
MAX_GEN = 100           # Generasi Maksimum
P_CROSSOVER = 0.8       # Probabilitas Crossover
P_MUTATION = 0.05       # Probabilitas Mutasi
N_FEATURES = 30         # Panjang Kromosom (30 gen) 

class GeneticAlgorithmFeatureSelection:
    def __init__(self, X, y, classifier):
        self.X = X
        self.y = y
        self.classifier = classifier

    def create_individual(self):
        # Binary Encoding: 0 (fitur dibuang), 1 (fitur terpilih)
        return np.random.randint(0, 2, N_FEATURES)

    def calculate_fitness(self, individual):
        # Jika tidak ada fitur terpilih, fitness = 0
        if np.sum(individual) == 0:
            return 0.0
        
        # Seleksi subset fitur berdasarkan kromosom 
        selected_indices = [i for i, bit in enumerate(individual) if bit == 1]
        X_subset = self.X[:, selected_indices]
        
        # Evaluasi menggunakan 10-Fold Cross Validation 
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        scores = cross_val_score(self.classifier, X_subset, self.y, cv=cv, scoring='accuracy')
        return scores.mean()

    def selection(self, population, fitness_scores):
        # Tournament Selection (Size=3) 
        tournament_size = 3
        selected = []
        for _ in range(len(population)):
            candidates_idx = np.random.choice(len(population), tournament_size, replace=False)
            best_idx = candidates_idx[np.argmax(fitness_scores[candidates_idx])]
            selected.append(population[best_idx])
        return np.array(selected)

    def crossover(self, parent1, parent2):
        # Single-point Crossover
        if np.random.rand() < P_CROSSOVER:
            point = np.random.randint(1, N_FEATURES - 1)
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
            return child1, child2
        return parent1.copy(), parent2.copy()

    def mutation(self, individual):
        # Bit-flip Mutation 
        for i in range(N_FEATURES):
            if np.random.rand() < P_MUTATION:
                individual[i] = 1 - individual[i]
        return individual

    def run(self):
        # Inisialisasi Populasi Awal 
        population = np.array([self.create_individual() for _ in range(POP_SIZE)])
        global_best_fitness = 0
        global_best_ind = None
        history = []

        print("\nMemulai Proses Evolusi...")
        for gen in range(MAX_GEN):
            fitness_scores = np.array([self.calculate_fitness(ind) for ind in population])
            
            # Elitisme: Simpan solusi terbaik sejauh ini 
            current_best_idx = np.argmax(fitness_scores)
            if fitness_scores[current_best_idx] > global_best_fitness:
                global_best_fitness = fitness_scores[current_best_idx]
                global_best_ind = population[current_best_idx].copy()
            
            history.append(global_best_fitness)
            
            if gen % 10 == 0:
                print(f"Generasi {gen}: Akurasi Terbaik = {global_best_fitness:.4f}")

            # Pembentukan Generasi Baru
            parents = self.selection(population, fitness_scores)
            next_gen = [global_best_ind] # Masukkan elit
            
            while len(next_gen) < POP_SIZE:
                p1, p2 = parents[np.random.randint(0, len(parents), 2)]
                c1, c2 = self.crossover(p1, p2)
                next_gen.append(self.mutation(c1))
                if len(next_gen) < POP_SIZE:
                    next_gen.append(self.mutation(c2))
            
            population = np.array(next_gen)

        return global_best_ind, global_best_fitness, history


# 4. EKSEKUSI & ANALISIS HASIL
# Menggunakan SVM (Kernel Linear) sebagai Wrapper
model_svm = SVC(kernel='linear', random_state=42)
ga = GeneticAlgorithmFeatureSelection(X_scaled, y, model_svm)
best_chromosome, best_accuracy, fitness_history = ga.run()

# Identifikasi Nama Fitur
selected_features = [feature_names[i] for i, bit in enumerate(best_chromosome) if bit == 1]
discarded_features = [feature_names[i] for i, bit in enumerate(best_chromosome) if bit == 0]

print("\n" + "="*50)
print("HASIL AKHIR OPTIMASI GENETIC ALGORITHM")
print("="*50)
print(f"Akurasi Final Terbaik   : {best_accuracy * 100:.2f}%")
print(f"Jumlah Fitur Terpilih   : {len(selected_features)} dari 30 fitur")
print(f"Rasio Reduksi           : {((30 - len(selected_features))/30)*100:.2f}%")

print("\n--- DAFTAR FITUR TERPILIH (RELEVAN) ---")
for i, name in enumerate(selected_features, 1):
    print(f"{i}. {name}")


# 5. VISUALISASI KONVERGENSI (Bab 5)
plt.figure(figsize=(10, 6))
plt.plot(fitness_history, color='teal', linewidth=2, label='Best Fitness (Accuracy)')
plt.title('Grafik Konvergensi Akurasi Genetic Algorithm', fontsize=14)
plt.xlabel('Generasi', fontsize=12)
plt.ylabel('Akurasi (Wrapper SVM)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()