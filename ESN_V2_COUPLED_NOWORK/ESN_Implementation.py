import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Windows: prevent OpenMP conflict between PyTorch and NumPy
from LT_data import load_and_transform, torch_data
from ESN_Architecture import EchoStateNetwork
from TEP_data import inverse_transform, calculate_nmse, calculate_mae, calculate_rmse, plot_results



def run_esn(INPUT_COLS, TARGET_COL, TRAIN_FRAC = 0.8, WASHOUT = 200, 
            plot = True, DATA_PATH  = "datasets/CESMDataset_Try.csv", #Hyperparams
            ESN_CONFIG = dict(n_reservoir = 500, spectral_radius = 0.9, sparsity = 0.9, 
                              leaking_rate = 0.3, input_scaling = 1.0, ridge_alpha = 1e-6)):

    df, stats = load_and_transform(DATA_PATH, INPUT_COLS + [TARGET_COL])
    U_train, U_test, Y_train, Y_test = torch_data(df, INPUT_COLS, TARGET_COL, TRAIN_FRAC)
    esn = EchoStateNetwork(n_inputs = len(INPUT_COLS), n_outputs = 1, **ESN_CONFIG)
    esn.fit(U_train, Y_train, washout = WASHOUT)
    Y_pred = esn.predict(U_test, washout = 0)
    Y_pred_raw = inverse_transform(Y_pred, stats, TARGET_COL).squeeze()
    Y_test_raw = inverse_transform(Y_test, stats, TARGET_COL).squeeze()
    metrics = {"nmse": calculate_nmse(Y_pred, Y_test), "rmse": calculate_rmse(Y_pred_raw, Y_test_raw), "mae":  calculate_mae(Y_pred_raw,  Y_test_raw)}
    print(metrics)
    if plot == True:
        plot_results(Y_pred_raw, Y_test_raw, TARGET_COL, metrics)
    return esn, metrics

run_esn(["PD_200m", "AMOC"], "SFWF")