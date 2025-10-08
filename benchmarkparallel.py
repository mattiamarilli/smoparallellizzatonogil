import time
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from smoparallel import SVM

def benchmark_svm(num_samples=1000, num_features=200, test_size=0.2, random_state=42):
    X, y = make_classification(
        n_samples=num_samples,
        n_features=num_features,
        n_informative=int(num_features * 0.6),
        n_redundant=int(num_features * 0.1),
        n_classes=2,
        n_clusters_per_class=1,
        random_state=random_state
    )
    y = 2 * y - 1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    gamma = 1.0 / num_features

    print("===================================")
    print("Benchmarking SVM Custom (1 thread)")
    print("===================================")
    model_single = SVM(
        numthreads=1,
        c=1.0,
        kkt_thr=1e-3,
        max_iter=1000,
        kernel_type='rbf',
        gamma_rbf=gamma
    )

    start_time = time.time()
    model_single.fit(X_train, y_train)
    training_time_single = time.time() - start_time
    print(f"Training time (1 thread): {training_time_single:.3f} s")
    print(f"Tempo colonne (1 thread): {model_single.sum_columns_calculation_time:.3f} s")

    print("\n===================================")
    print("Benchmarking SVM Custom (multi-thread)")
    print("===================================")
    num_threads = 2
    model_multi = SVM(
        numthreads=num_threads,
        c=1.0,
        kkt_thr=1e-3,
        max_iter=1000,
        kernel_type='rbf',
        gamma_rbf=gamma
    )

    start_time = time.time()
    model_multi.fit(X_train, y_train)
    training_time_multi = time.time() - start_time
    print(f"Training time ({num_threads} threads): {training_time_multi:.3f} s")
    print(f"Tempo colonne ({num_threads} threads): {model_multi.sum_columns_calculation_time:.3f} s")

    # --- Calcolo speedup ed efficiency SOLO sulla parte parallelizzata ---
    column_speedup = model_single.sum_columns_calculation_time / model_multi.sum_columns_calculation_time
    column_efficiency = column_speedup / num_threads

    print("\n-----------------------------------")
    print(f"Speedup colonne: {column_speedup:.2f}x")
    print(f"Efficiency colonne: {column_efficiency * 100:.1f}%")
    print("-----------------------------------")

    # --- Valutazione accuratezza ---
    start_time = time.time()
    y_pred, scores = model_multi.predict(X_test)
    prediction_time = time.time() - start_time
    accuracy = np.mean(y_pred == y_test)
    print(f"\nPrediction time: {prediction_time:.3f} s")
    print(f"Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    benchmark_svm()
