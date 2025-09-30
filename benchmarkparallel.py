import time
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from smoparallel import SVM

def benchmark_fit_multithreading(numthreads_list=(1,2,3,4), num_samples=10000, num_features=10):
    X, y = make_classification(
        n_samples=num_samples,
        n_features=num_features,
        n_informative=int(num_features*0.6),
        n_redundant=int(num_features*0.1),
        n_classes=2,
        random_state=42
    )
    y = 2*y - 1
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = StandardScaler().fit_transform(X_train)

    gamma = 1.0 / num_features
    results = {}

    for p in numthreads_list:
        model = SVM(numthreads=p, c=1.0, max_iter=1000, gamma_rbf=gamma)
        model.fit(X_train, y_train)
        results[p] = model.sumtimes_colum_calculation
        print(f"Threads={p} | Somma tempi colonne={model.sumtimes_colum_calculation:.4f} s")

    print("\n--- Speedup/Efficiency ---")
    T1 = results[1]
    for p, Tp in results.items():
        speedup = T1 / Tp
        efficiency = speedup / p
        print(f"Threads={p} | Time={Tp:.4f} s | Speedup={speedup:.2f} | Efficiency={efficiency:.2f}")


if __name__ == "__main__":
    benchmark_fit_multithreading()
