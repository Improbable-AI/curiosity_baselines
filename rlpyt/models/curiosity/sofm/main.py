import numpy as np
import sofm
import sys
sys.path.append("..")
import online_fuzzy_art as ofa
from time import time_ns
import matplotlib.pyplot as plt


if __name__ == "__main__":
    print("NB! Make sure to disable shuffling of lists in Python and C++ to be able to compare the results")

    rng = np.random.default_rng(12345)

    rho = 0.95
    beta = 0.01
    alpha = 0.1
    num_sims = 1
    num_batches = 15
    batch_size = 80
    num_features = 16


    time_cpp = np.empty((num_sims, num_batches))
    time_python = np.empty((num_sims, num_batches))

    for sim in range(num_sims):
        print(f"SIMULATION {sim}\n")
        features_batch = rng.uniform(size=(num_batches, batch_size, num_features))

        online_fuzzy_art_python = ofa.OnlineFuzzyART(rho, alpha, beta, num_features)
        online_fuzzy_art_cpp = sofm.art.OnlineFuzzyART(rho, alpha, beta, num_features)

        for batch, features in enumerate(features_batch):
            # np.savetxt(f"features_{k}.txt", features)
            print("C++ start")
            start = time_ns()
            clusters_cpp = np.array(online_fuzzy_art_cpp.run_online(features, max_epochs = 20))
            stop = time_ns()
            elapsed_ms = (stop - start) * 1e-6
            print(f"C++ took {elapsed_ms} ms")
            time_cpp[sim, batch] = elapsed_ms
            print("Python start")
            start = time_ns()
            clusters_python = np.array(online_fuzzy_art_python.run_online(features, [(0, 1)]*num_features, max_epochs = 20)[1])
            stop = time_ns()
            elapsed_ms = (stop - start) * 1e-6
            print(f"Python took {elapsed_ms} ms")
            time_python[sim, batch] = elapsed_ms
            assert (clusters_cpp == clusters_python).all(), f"unequal clusters at iteration {batch}:\nc++: {clusters_cpp}\nPython: {clusters_python}"

    print("No asserts - Test passed!")

    np.savetxt("timings_cpp_ms.txt", time_cpp)
    np.savetxt("timings_python_ms.txt", time_python)

    batch_numbers = np.arange(num_batches)

    avg_time_cpp_batches = time_cpp.mean(axis=0)
    avg_time_python_batches = time_python.mean(axis=0)

    std_time_cpp_batches = time_cpp.std(axis=0)
    std_time_python_batches = time_python.std(axis=0)

    plt.plot(avg_time_cpp_batches, 'b--', label='C++')
    plt.plot(avg_time_cpp_batches, 'bo')
    plt.fill_between(batch_numbers, avg_time_cpp_batches - std_time_cpp_batches, avg_time_cpp_batches + std_time_cpp_batches, color="blue", alpha=0.3)
    plt.plot(avg_time_python_batches, 'g--', label='Python')
    plt.plot(avg_time_python_batches, 'go')
    plt.fill_between(batch_numbers, avg_time_python_batches - std_time_python_batches, avg_time_python_batches + std_time_python_batches, color="green", alpha=0.3)
    plt.ylabel("Time [ms]")
    plt.xlabel("Batch number")
    plt.title(f"Timings of C++ vs Python over {num_sims} simulations with batch size {batch_size} and {num_features} features")
    plt.legend()
    plt.yscale("log")

    plt.show()