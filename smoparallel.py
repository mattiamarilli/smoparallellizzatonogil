from typing import Tuple
import numpy as np
from threading import Thread
import time

class SVM:
    def __init__(self, numthreads: int = 1, c: float = 1., kkt_thr: float = 1e-3, max_iter: int = 1e4, kernel_type: str = 'linear', gamma_rbf: float = 1.) -> None:
        if kernel_type not in ['linear', 'rbf']:
            raise ValueError('kernel_type must be either {} or {}'.format('linear', 'rbf'))

        super().__init__()

        self.c = float(c)
        self.max_iter = int(max_iter)
        self.kkt_thr = kkt_thr
        self.gamma_rbf = gamma_rbf
        self.b = 0.0
        self.alpha = np.array([])
        self.support_vectors = np.array([])
        self.support_labels = np.array([])
        self.numthreads = numthreads
        self.sumtimes_colum_calculation = 0

    # ------------------- PREDICT -------------------
    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.alpha.shape[0] == 0:
            raise ValueError("Model not trained yet.")
        K_test = self.kernel(self.support_vectors, x)
        scores = np.dot(self.alpha * self.support_labels, K_test) + self.b
        pred = np.sign(scores)
        return pred, scores

    def mvp_heuristic(self, error_cache: np.ndarray) -> Tuple[int, int]:
        alpha = self.alpha
        y = self.support_labels
        C = self.c

        L_indices = (alpha == 0)
        U_indices = (alpha == C)
        Free_indices = (alpha > 0) & (alpha < C)

        R_mask = (L_indices & (y == 1)) | (U_indices & (y == -1)) | Free_indices
        S_mask = (L_indices & (y == -1)) | (U_indices & (y == 1)) | Free_indices

        R_indices = np.where(R_mask)[0]
        S_indices = np.where(S_mask)[0]

        # filtrare fuori i NaN
        R_indices = [i for i in R_indices if not np.isnan(error_cache[i])]
        S_indices = [i for i in S_indices if not np.isnan(error_cache[i])]

        if len(R_indices) == 0 or len(S_indices) == 0:
            return -1, -1

        i_mvp = R_indices[np.argmin(error_cache[R_indices])]
        j_mvp = S_indices[np.argmax(error_cache[S_indices])]

        return i_mvp, j_mvp

    # ------------------- FIT -------------------
    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        N, D = x_train.shape
        self.b = 0.0
        self.alpha = np.zeros(N)
        self.support_labels = y_train
        self.support_vectors = x_train

        iter_idx = 0

        sq_norms = np.sum(x_train ** 2, axis=1)
        K = np.exp(-self.gamma_rbf * (sq_norms[:, None] + sq_norms[None, :] - 2 * x_train @ x_train.T))
        error_cache = (self.alpha * y_train) @ K + self.b - y_train

        print("SVM training using SMO algorithm - START")

        while iter_idx < self.max_iter:
            # if iter_idx % 50 == 0:
            #     for i in range(N):
            #         K_i = self.kernel_column(i)
            #         error_cache[i] = np.dot(self.alpha * self.support_labels, K_i) + self.b - y_train[i]

            i_2, i_1 = self.mvp_heuristic(error_cache)  # Seleziona i_MVP e j_MVP

            if i_2 == -1 or i_1 == -1:
                break

            if i_1 == i_2:
                print("Hello")
                continue

            y_1, alpha_1 = self.support_labels[i_1], self.alpha[i_1]
            y_2, alpha_2 = self.support_labels[i_2], self.alpha[i_2]

            start_time = time.time()
            K_i1 = self.rbf_kernel_column_multithread(i_1)
            K_i2 = self.rbf_kernel_column_multithread(i_2)
            self.sumtimes_colum_calculation+= time.time() - start_time
            k11 = K_i1[i_1]
            k22 = K_i2[i_2]
            k12 = K_i1[i_2]

            # --- Calcolo boundaries ---
            L, H = self.compute_boundaries(alpha_1, alpha_2, y_1, y_2)

            # --- Calcolo eta ---
            eta = k11 + k22 - 2 * k12
            if eta < 1e-12:
                continue

            # --- Calcolo errori ---
            E_1 = np.dot(self.alpha * self.support_labels, K_i1) + self.b - y_1
            E_2 = np.dot(self.alpha * self.support_labels, K_i2) + self.b - y_2

            # --- Aggiornamento alpha ---
            alpha_2_new = alpha_2 + y_2 * (E_1 - E_2) / eta
            alpha_2_new = np.clip(alpha_2_new, L, H)
            alpha_1_new = alpha_1 + y_1 * y_2 * (alpha_2 - alpha_2_new)

            # --- Aggiornamento b ---
            b1 = self.b - E_1 - y_1 * (alpha_1_new - alpha_1) * k11 - y_2 * (alpha_2_new - alpha_2) * k12
            b2 = self.b - E_2 - y_1 * (alpha_1_new - alpha_1) * k12 - y_2 * (alpha_2_new - alpha_2) * k22

            if 0 < alpha_1_new < self.c:
                self.b = b1
            elif 0 < alpha_2_new < self.c:
                self.b = b2
            else:
                self.b = (b1 + b2) / 2

            # --- Salvataggio alpha aggiornati ---
            self.alpha[i_1] = alpha_1_new
            self.alpha[i_2] = alpha_2_new

            # --- Aggiornamento cache errori ---
            delta_alpha_1 = alpha_1_new - alpha_1
            delta_alpha_2 = alpha_2_new - alpha_2
            error_cache += y_1 * delta_alpha_1 * K_i1 + y_2 * delta_alpha_2 * K_i2

            iter_idx += 1

        # --- Filtraggio support vectors ---
        support_vectors_idx = (self.alpha != 0)
        self.support_labels = self.support_labels[support_vectors_idx]
        self.support_vectors = self.support_vectors[support_vectors_idx, :]
        self.alpha = self.alpha[support_vectors_idx]

        print("SVM training using SMO algorithm - DONE!")

    # ------------------- BOUNDS -------------------
    def compute_boundaries(self, alpha_1, alpha_2, y_1, y_2) -> Tuple[float, float]:
        if y_1 == y_2:
            lb = max(0, alpha_1 + alpha_2 - self.c)
            ub = min(self.c, alpha_1 + alpha_2)
        else:
            lb = max(0, alpha_2 - alpha_1)
            ub = min(self.c, self.c + alpha_2 - alpha_1)
        return lb, ub

    def rbf_kernel_column_multithread(self, i: int) -> np.ndarray:
        n = self.support_vectors.shape[0]
        x_i = self.support_vectors[i]
        n_threads = self.numthreads
        col = np.zeros(n)

        # Dividi gli indici tra i thread
        batch_size = n // n_threads
        threads = []
        for t in range(n_threads):
            start_idx = t * batch_size
            end_idx = n if t == n_threads - 1 else (t + 1) * batch_size
            thread = Thread(target=compute_rbf_block,
                            args=(self.support_vectors, x_i, self.gamma_rbf,
                                  list(range(start_idx, end_idx)), col, start_idx))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        return col

def compute_rbf_block(support_vectors, x_i, gamma_rbf, indices, result_list, offset):
    for idx, global_idx in enumerate(indices):
        diff = support_vectors[global_idx] - x_i
        result_list[offset + idx] = np.exp(-gamma_rbf * np.dot(diff, diff))