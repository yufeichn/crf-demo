import time
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
from sklearn.svm import SVC
from pyriemann.utils.mean import mean_euclid
from scipy.linalg import fractional_matrix_power

def train_svm(X_train, y_train):
    X_train = X_train.reshape(X_train.shape[0], -1)
    clf = SVC(kernel='linear')
    clf.fit(X_train, y_train)
    return clf

def train_lda(X_train, y_train):
    X_train = X_train.reshape(X_train.shape[0], -1)
    clf = LinearDiscriminantAnalysis()
    clf.fit(X_train, y_train)
    return clf

def whitening_transformation_train(X_train):
    cov_train = np.array([np.cov(x) for x in X_train])
    mean_cov = mean_euclid(cov_train)
    whitening_matrix = fractional_matrix_power(mean_cov, -0.5)
    whitened_X_train = np.array([np.dot(whitening_matrix, x) for x in X_train])
    return whitened_X_train, whitening_matrix

def whitening_transformation_test(X_test, whitening_matrix):
    if X_test.ndim == 2:
        X_test = np.expand_dims(X_test, axis=0)
    whitened_X_test = np.array([np.dot(whitening_matrix, x) for x in X_test])
    return whitened_X_test

def compute_regularized_cov(x, reg=1e-8):
    cov = np.cov(x)
    cov_reg = cov + reg * np.eye(cov.shape[0])
    return cov_reg

def whitening_transformation_train_reg(X_train, reg=1e-8):
    cov_train = np.array([compute_regularized_cov(x, reg) for x in X_train])
    mean_cov = np.mean(cov_train, axis=0)
    mean_cov_reg = mean_cov + reg * np.eye(mean_cov.shape[0])
    whitening_matrix = fractional_matrix_power(mean_cov_reg, -0.5)
    whitened_X_train = np.array([np.dot(whitening_matrix, x) for x in X_train])
    return whitened_X_train, whitening_matrix

def test_single(x, y, clf):
    if x.ndim == 2:
        x = np.expand_dims(x, axis=0)
    x_reshaped = x.reshape(x.shape[0], -1)
    y_pred = clf.predict(x_reshaped)[0]
    print(f"Predict Label = {y_pred}, True Label = {y}")
    return y_pred

def update_realtime_accuracy(total, correct, y_true, y_pred):
    total += 1
    if y_true == y_pred:
        correct += 1
    running_acc = correct / total
    print(f"Mean ACC: {running_acc:.4f} ({correct}/{total})")
    return total, correct



if __name__ == "__main__":
    # ======================================== Dataloader ===================================================
    '''
    X_train = [60, 18, 500] → [Batch_size, Num_EEG_Channels, Num_Samples]
    y_train = [60,] → [Batch_size], 0 → Left, 1 → Right
    X_test = [60, 18, 500] → [Batch_size, Num_EEG_Channels, Num_Samples]
    y_test = [60,] → [Batch_size], 0 → Left, 1 → Right
    '''
    X_train = np.load('X_train.npy')
    Y_train = np.load('Y_train.npy')
    X_test = np.load('X_test.npy')
    Y_test = np.load('Y_test.npy')
    test_num = len(Y_test)

    # ================================= 从此处开始需要与语音结合 ===================================================
    # Step1: Calibration & Train the model
    print("Start Training...")
    X_train, W_Matrix = whitening_transformation_train_reg(X_train, reg=1e-8)  # Data Calibration Algorithm
    model_weights = train_svm(X_train, Y_train)  # Train Linear Model

    # Step2: Test Stage, 模拟真实数据分发，数据将按 1s 连续输入系统中
    running_total = 0
    running_correct = 0
    all_predictions = []  # 存储所有预测结果

    print("Start Testing...")
    for i in range(0, test_num):
        # time.sleep(1)
        # 以下针对每1s的数据，都会重复进行，实时输出预测结果
        x_test = X_test[i, :, :]  # x_test = [18, 500] → 1s EEG Data
        y_test = Y_test[i]  # y_test = 0 or 1, 0 → Left, 1 → Right

        x_test = whitening_transformation_test(x_test, W_Matrix)  # Data Calibration Algorithm

        y_pred = test_single(x_test, y_test, model_weights)  # Test model
        all_predictions.append(y_pred)  # 记录预测结果
        running_total, running_correct = update_realtime_accuracy(running_total, running_correct, y_test, y_pred)

    # 计算总体准确率
    all_predictions = np.array(all_predictions)
    total_accuracy = np.mean(all_predictions == Y_test)
    print("\n==================================================")
    print(f"总体准确率评估: {total_accuracy:.4f} ({np.sum(all_predictions == Y_test)}/{len(Y_test)})")
    print("==================================================")
