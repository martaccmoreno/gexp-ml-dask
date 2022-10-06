import gc
import csv
import time
import timeit

import matplotlib.pyplot as plt
import pandas as pd
from dask.distributed import Client
from memory_profiler import memory_usage

from pipelines.dask_pipeline import dask_pipeline
from pipelines.spe_pipeline import spe_pipeline

if __name__ == '__main__':
    prediction_type = {'BRCA': 'classification', 'BRCA_coding': 'classification', 'LUAD/LUSC': 'regression',
                       'SYNTH': 'classification'}

    # file path format: (features, labels)
    benchmark_dask = {
        'BRCA': ('data/brca_data/brca_fpkm.parquet', 'data/brca_data/brca_subtypes.parquet'),
        'BRCA_coding': ('data/brca_data/coding_brca_fpkm.parquet', 'data/brca_data/brca_subtypes.parquet'),
        'LUAD/LUSC': ('data/lung_data/coding_lung_fpkm.parquet', 'data/lung_data/coding_cigs_per_day.parquet'),
        'SYNTH': ('data/synthetic_data/shuffled/n5000_f20000_synthetic_features.parquet',
                  'data/synthetic_data/shuffled/n5000_synthetic_labels.parquet')
    }

    benchmark_spe = {
        'BRCA': ('data/brca_data/brca_fpkm.csv', 'data/brca_data/brca_subtypes.csv'),
        'BRCA_coding': ('data/brca_data/coding_brca_fpkm.csv', 'data/brca_data/brca_subtypes.csv'),
        'LUAD/LUSC': ('data/lung_data/coding_lung_fpkm.csv', 'data/lung_data/cigs_per_day.csv'),
        'SYNTH': ('data/synthetic_data/shuffled/n5000_f20000_synthetic_features.csv',
                  'data/synthetic_data/shuffled/n5000_synthetic_labels.csv')
    }

    peak_mem_dict = {'BRCA': {}, 'BRCA_coding': {}, 'LUAD/LUSC': {}, 'SYNTH': {}}
    runtime_dict = {'BRCA': {}, 'BRCA_coding': {}, 'LUAD/LUSC': {}, 'SYNTH': {}}
    cv_scores_dict = {'BRCA': {}, 'BRCA_coding': {}, 'LUAD/LUSC': {}, 'SYNTH': {}}
    eval_scores_dict = {'BRCA': {}, 'BRCA_coding': {}, 'LUAD/LUSC': {}, 'SYNTH': {}}

    for dataset, fps in benchmark_dask.items():
        for n_workers, fw in [(1, 'Dask (Threaded)'), (8, 'Dask (Distributed)')]:
            client = Client(n_workers=n_workers, silence_logs=70)
            print(fw)
            print(client)

            task = prediction_type[dataset]

            print(f"Benchmarking {dataset} on {fw}...")
            print("PEAK MEMORY CONSUMPTION")
            peak_mem, retval = memory_usage([dask_pipeline, (fps[0], fps[1], task, None)], max_usage=True,
                                            include_children=True, retval=True)
            mean_cv_score, std_cv_score, eval_score = retval
            print(f"Peak memory usage: {peak_mem}")
            print(f"CV score: {mean_cv_score} +- {std_cv_score}")
            print(f"Evaluation score: {eval_score}")

            peak_mem_dict[dataset].update({fw: peak_mem})
            cv_scores_dict[dataset].update({fw: f"{mean_cv_score} +- {std_cv_score}"})
            eval_scores_dict[dataset].update({fw: eval_score})

            client.restart()
            gc.collect()

            print("MINIMUM RUNTIME")
            # 3 repetitions to determine minimum runtime
            runtime = timeit.repeat(f"{dask_pipeline.__name__}(fps[0], fps[1], task, None)",
                                    setup=f"from __main__ import {dask_pipeline.__name__}, fps, task", repeat=3,
                                    number=1)
            min_runtime = min(runtime)
            print(f"Minimum runtime: {min_runtime}")

            runtime_dict[dataset].update({fw: min_runtime})

            client.shutdown()
            gc.collect()
            time.sleep(60)

    for dataset, fps in benchmark_spe.items():
        fw = 'Scientific Python Environment'
        task = prediction_type[dataset]

        print(f"Benchmarking {dataset} on {fw}...")

        print("PEAK MEMORY CONSUMPTION")
        peak_mem, retval = memory_usage([spe_pipeline, (fps[0], fps[1], task, None)], max_usage=True,
                                        include_children=True, retval=True)
        mean_cv_score, std_cv_score, eval_score = retval
        print(f"Peak memory usage: {peak_mem}")
        print(f"CV score: {mean_cv_score} +- {std_cv_score}")
        print(f"Evaluation score: {eval_score}")

        peak_mem_dict[dataset].update({fw: peak_mem})
        cv_scores_dict[dataset].update({fw: f"{mean_cv_score} +- {std_cv_score}"})
        eval_scores_dict[dataset].update({fw: eval_score})

        print("MINIMUM RUNTIME")
        runtime = timeit.repeat(f"{spe_pipeline.__name__}(fps[0], fps[1], task, None)",
                                setup=f"from __main__ import {spe_pipeline.__name__}, fps, task", repeat=3,
                                number=1)
        min_runtime = min(runtime)
        print(f"Minimum runtime: {min_runtime}")

        runtime_dict[dataset].update({fw: min_runtime})

        print(peak_mem_dict)
        print(runtime_dict)
        print(cv_scores_dict)
        print(eval_scores_dict)

        gc.collect()

    with open("benchmark_results/default_peakmem_benchmark.csv", "w") as outfile:
        w_mem = csv.writer(outfile)
        w_mem.writerow(['Dimension', 'Framework', "Peak Memory (MiB)"])
        for key, val in peak_mem_dict.items():
            for subkey, subval in val.items():
                w_mem.writerow([key, subkey, subval])

    with open("benchmark_results/default_runtime_benchmark.csv", "w") as outfile:
        w_time = csv.writer(outfile)
        w_time.writerow(['Dimension', 'Framework', "Fastest Runtime (s)"])
        for key, val in runtime_dict.items():
            for subkey, subval in val.items():
                w_time.writerow([key, subkey, subval])

    with open("benchmark_results/default_evalscore_benchmark.csv", "w") as outfile:
        w_eval = csv.writer(outfile)
        w_eval.writerow(['Dimension', 'Framework', "Evaluation Score (Accuracy)"])
        for key, val in eval_scores_dict.items():
            for subkey, subval in val.items():
                w_eval.writerow([key, subkey, subval])

    pd.DataFrame(runtime_dict).transpose().plot(kind='bar', color=["steelblue", "deepskyblue", "wheat"])
    plt.title('Minimum runtime for different machine learning frameworks')
    plt.xticks(rotation=360)
    plt.ylabel('Minimum Runtime (s)')
    plt.xlabel('Datasets')
    plt.savefig("benchmark_results/default_runtime_benchmark.pdf")

    pd.DataFrame(peak_mem_dict).transpose().plot(kind='bar', color=["steelblue", "deepskyblue", "wheat"])
    plt.title('Peak memory consumption for different machine learning frameworks')
    plt.xticks(rotation=360)
    plt.ylabel('Peak Memory Consumption (MiB)')
    plt.xlabel('Datasets')
    plt.savefig("benchmark_results/default_peakmem_benchmark.pdf")

    pd.DataFrame(eval_scores_dict).transpose().plot(kind='bar', color=["steelblue", "deepskyblue", "wheat"])
    plt.title('Evaluation accuracy scores for different machine learning frameworks')
    plt.xticks(rotation=360)
    plt.ylabel('Accuracy')
    plt.xlabel('Datasets')
    plt.savefig("benchmark_results/default_evalscore_benchmark.pdf")
