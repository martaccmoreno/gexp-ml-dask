import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

##################
# PIPELINE PLOTS #
##################

dim_runtime_compare = {'n40kxf10k (1.8GB)': {'Distributed Threads': 51.513321309001185,
                                             'Distributed Processes': 31.649477670000124,
                                             'Scientific Python Environment': 7.243791699998837},
                       'n60kxf10k (2.7GB)': {'Distributed Threads': 72.87259176399675,
                                             'Distributed Processes': 45.06167149900102,
                                             'Scientific Python Environment': 12.208875541000452},
                       'n87.9kxf10k (3.9GB)': {'Distributed Threads': 105.24801276699873,
                                               'Distributed Processes': 62.1773908920004,
                                               'Scientific Python Environment': 43.652458317999844},
                        'n40kxf15k (2.7GB)': {'Distributed Threads': 73.33812886400847,
                                              'Distributed Processes': 51.099937504999616,
                                              'Scientific Python Environment': 12.660397250001552},
                       'n60kxf15k (4.0GB)': {'Distributed Threads': 105.68495018497924,
                                             'Distributed Processes': 69.0846751579993,
                                             'Scientific Python Environment': 49.173891313999775},
                       'n87.9kxf15k (5.9GB)': {'Distributed Threads': 149.57372026797384,
                                               'Distributed Processes': 97.57314335100091,
                                               'Scientific Python Environment': 210.00784203700096},
                       'n40kxf20k (3.6GB)': {'Distributed Threads': 100.2008771779947,
                                             'Distributed Processes': 71.48992498299776,
                                             'Scientific Python Environment': 37.45534433300054},
                       'n60kxf20k (5.3GB)': {'Distributed Threads': 146.63784548299736,
                                             'Distributed Processes': 99.47603979199994,
                                             'Scientific Python Environment': 188.27392202400006},
                       'n87.9kxf20k (7.8GB)': {'Distributed Threads': 204.43513365398394,
                                               'Distributed Processes': 137.16863535200173},
                       'n40kxf24.2k (4.3GB)': {'Distributed Threads': 130.91702970201732,
                                               'Distributed Processes': 92.9625184209981},
                       'n60kxf24.2k (6.5GB)': {'Distributed Threads': 188.6701306469913,
                                               'Distributed Processes': 130.3858079490019},
                       'n87.9kxf24.2k (9.5GB)': {'Distributed Threads': 268.80597571100225,
                                                 'Distributed Processes': 182.53899199499938}}

matplotlib.rc('xtick', labelsize=16)
matplotlib.rc('ytick', labelsize=16)
params = {'legend.fontsize': 18, 'legend.handlelength': 1.5}
plt.rcParams.update(params)

fig, axes = plt.subplots(figsize=(13, 16), dpi=250, nrows=2, ncols=1, gridspec_kw={'height_ratios': [4, 1]})
fig.subplots_adjust(top=0.8, bottom=0.1, left=0.1, right=0.95)

pd.DataFrame(dim_runtime_compare).transpose().iloc[::-1].plot(kind='barh', color=["steelblue", "deepskyblue", "wheat"],
                                                              ax=axes[0])
axes[0].set_title('A. Runtime for preprocessing sampled SC datasets\n', fontdict={'fontsize': 22})
plt.xticks(rotation=45)
axes[0].set_xlabel('Minimum Runtime (s)\n', fontdict={'fontsize': 18})
axes[0].set_ylabel('Dataset Dimensions', fontdict={'fontsize': 18})
axes[0].text(1, 0.05, "*", fontsize=20, color="wheat", fontweight="bold")
axes[0].text(1, 1.05, "*", fontsize=20, color="wheat", fontweight="bold")
axes[0].text(1, 2.05, "*", fontsize=20, color="wheat", fontweight="bold")
axes[0].text(1, 3.05, "*", fontsize=20, color="wheat", fontweight="bold")

#############
# DASK LOAD #
#############

dim_load_time = {
    'n40k x f10k (1.8GB)': {'Distributed Threads': 18.87808121999842, 'Distributed Processes': 12.17565520197968},
    'n60k x f10k (2.7GB)': {'Distributed Threads': 28.13538479898125, 'Distributed Processes': 17.371184975025244},
    'n40k x f15k (2.7GB)': {'Distributed Threads': 38.69335266499547, 'Distributed Processes': 22.614442268997664},
    'n60k x f15k (4.0GB)': {'Distributed Threads': 62.871802929992555, 'Distributed Processes': 38.101147094974294}}

pd.DataFrame(dim_load_time).transpose().iloc[::-1].plot(kind='barh', color=["steelblue", "deepskyblue"],
                                                        ax=axes[1])
axes[1].set_title('B. Runtime for ' + r"$\bf{full}$" + ' loading sampled SC datasets\n', fontdict={'fontsize': 22})
axes[1].set_xlabel('Minimum Runtime (s)', fontdict={'fontsize': 18})
axes[1].set_ylabel('Dataset Dimensions', fontdict={'fontsize': 18})
plt.tight_layout()

###################
# SAVE BOTH PLOTS #
###################

plt.savefig(f'results/sc_runtime.pdf')
