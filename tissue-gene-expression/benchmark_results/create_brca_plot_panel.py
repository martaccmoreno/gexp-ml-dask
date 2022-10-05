import matplotlib.pyplot as plt
import pandas as pd


def results_csv_to_dic(filename):
    dic = {}
    with open(filename) as f:
        lines = f.readlines()
        for line in lines[1:]:
            line = line.strip()
            n, fw, val = line.split(',')
            if n in dic:
                dic[n].update({fw: float(val)})
            else:
                dic[n] = {fw: float(val)}
    return dic


samplewise_peakmem = results_csv_to_dic(
    "samplewise/brca_undersample_peakmem_benchmark_samplewise.csv")
samplewise_runtime = results_csv_to_dic(
    "samplewise/brca_undersample_runtime_benchmark_samplewise.csv")
samplewise_evalscore = results_csv_to_dic(
    "samplewise/brca_undersample_evalscore_benchmark_samplewise.csv")

featurewise_peakmem = results_csv_to_dic(
    "featurewise/brca_undersample_peakmem_benchmark_featurewise.csv")
featurewise_runtime = results_csv_to_dic(
    "featurewise/brca_undersample_runtime_benchmark_featurewise.csv")
featurewise_evalscore = results_csv_to_dic(
    "featurewise/brca_undersample_evalscore_benchmark_featurewise.csv")

params = {'legend.fontsize': 16, 'legend.handlelength': 1.2}
plt.rcParams.update(params)

fig = plt.figure(figsize=(25, 16), dpi=250)
fig.subplots_adjust(top=0.8, bottom=0.1, left=0.1, right=0.95)
subfigs = fig.subfigures(2, 1)

#########
# Row 1 #
#########
subfigs[0].suptitle('A. Sample-wise Subsampling with n Samples (f features = 20,000)', x=0.325, y=0.95, fontsize=24)
axes0 = subfigs[0].subplots(nrows=1, ncols=3)

# peakmem col 1
bars = pd.DataFrame(samplewise_peakmem).transpose().plot(kind='bar', color=["steelblue", "deepskyblue", "wheat"],
                                                  ax=axes0[0], legend=False, rot=360, fontsize=16, ylim=(0, 10000))
axes0[0].set_title('Peak Memory Consumption (MiB)', fontsize=20)
axes0[0].text(x=1.42, y=1600,
              s='n200 = 54.6 MB\nn600 = 161.3 MB\nn1205 = 327.3 MB',
              size=16, ha='left', va='top',
              bbox=dict(ec='lightgrey', fc='w', alpha=0.8))
axes0[0].legend(loc="upper left")

# runtime col 2
pd.DataFrame(samplewise_runtime).transpose().plot(kind='bar', color=["steelblue", "deepskyblue", "wheat"],
                                                  ax=axes0[1], legend=False, rot=360, fontsize=16, ylim=(0, 2250))
axes0[1].set_title('Minimum Runtime (s)', fontsize=20)
axes0[1].legend(loc="upper left")

# evalscore col 3
pd.DataFrame(samplewise_evalscore).transpose().plot(kind='bar', color=["steelblue", "deepskyblue", "wheat"],
                                                  ax=axes0[2], legend=False, rot=360, fontsize=16, ylim=(0, 1))
axes0[2].set_title('Evaluation Accuracy', fontsize=20)
axes0[2].legend(loc="lower left")

#########
# Row 2 #
#########
subfigs[1].suptitle('B. Feature-wise Subsampling with f Features (n samples = 1,205)', y=0.95, x=0.325, fontsize=24)
axes1 = subfigs[1].subplots(nrows=1, ncols=3)

# peakmem col 1
pd.DataFrame(featurewise_peakmem).transpose().plot(kind='bar', color=["steelblue", "deepskyblue", "wheat"],
                                                  ax=axes1[0], legend=True, rot=360, fontsize=16, ylim=(0, 10000))
axes1[0].set_title('Peak Memory Consumption (MiB)', fontsize=20)
axes1[0].text(x=1.35, y=1600,
              s='f10,000 = 165.3 MB\nf20,000 = 327.3 MB\nf40,000 = 577.4 MB',
              size=16, ha='left', va='top',
              bbox=dict(ec='lightgrey', fc='w', alpha=0.8))

# runtime col 2
pd.DataFrame(featurewise_runtime).transpose().plot(kind='bar', color=["steelblue", "deepskyblue", "wheat"],
                                                  ax=axes1[1], legend=True, rot=360, fontsize=16, ylim=(0, 2250))
axes1[1].set_title('Minimum Runtime (s)', fontsize=20)

# evalscore col 3
pd.DataFrame(featurewise_evalscore).transpose().plot(kind='bar', color=["steelblue", "deepskyblue", "wheat"],
                                                    ax=axes1[2], legend=False, rot=360, fontsize=16, ylim=(0, 1))
axes1[2].set_title('Evaluation Accuracy', fontsize=20)
axes1[2].legend(loc="lower left")

plt.savefig('brca_subsample_panel.pdf')
