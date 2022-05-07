import numpy as np
import art
import sys
sys.path.append("..")
import online_fuzzy_art as ofa

if __name__ == "__main__":
    rng = np.random.default_rng(12345)

    rho = 0.95
    beta = 0.01
    alpha = 0.1

    features_batch = rng.uniform(size=(15, 5, 5))
    num_features = features_batch.shape[2]

    online_fuzzy_art = art.OnlineFuzzyART(rho, alpha, beta, num_features)
    online_fuzzy_art_python = ofa.OnlineFuzzyART(rho, alpha, beta, num_features)

    for k, features in enumerate(features_batch):
        print("Start of for looop!")
        np.savetxt(f"features_{k}.txt", features)
        clusters = np.array(online_fuzzy_art.run_online(features))
        clusters_python = np.array(online_fuzzy_art_python.run_online(features, [(0, 1)]*num_features)[1])
        assert (clusters == clusters_python).all(), f"unequal clusters at iteration {k}:\nc++: {clusters}\nPython: {clusters_python}"
        print("End of for looop!")