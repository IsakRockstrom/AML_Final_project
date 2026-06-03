from itertools import combinations
from ESN_Implementation import run_esn
import numpy as np

def combinator_features(all_features = ["PD_200m", "SFWF", "ICEFRAC", "PD_0m", "SALT_500m", "NAO", "TAUX", "N_SALT", "AABW"],
                        current_error_sum = 1000, TARGET_COL = "AMOC"):
    
    number_features = np.arange(2, len(all_features) + 1)
    final_results = []
    for n_features in number_features:
        for comb in combinations(all_features, n_features):
            INPUT_COLS = list(comb)
            _, metrics = run_esn(INPUT_COLS, TARGET_COL, plot = False)
            nmse, rmse, mae = metrics["nmse"], metrics["rmse"], metrics["mae"]
            tested_comb = INPUT_COLS + [nmse, rmse, mae]
            error_sum = nmse + rmse + mae
            if error_sum < current_error_sum:
                best_features = INPUT_COLS
                current_error_sum = error_sum
            final_results.append(tested_comb)
    
    print(best_features)
    
    file_path = 'Combinator_results.txt'
    with open(file_path, 'w') as f:
        for sublist in final_results:
            f.write(' '.join(map(str, sublist)) + '\n')

combinator_features()