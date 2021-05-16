import numpy as np
import matplotlib
from matplotlib import gridspec
import matplotlib.pyplot as plt
import torch
import pdb

# Initialize
font = {'size'   : 5}
matplotlib.rc('font', **font)

attention_type = 'effective'
datasets = ['RTE', 'MRPC', 'QNLI', 'SST-2', 'STS-B']
# datasets = ['RTE']
tags = ["NOUN", "PRON", "VERB", "[SEP]", "[CLS]", "PUNCT"]

# Read data
ling_feature_att = {}
global_max = 0
global_min = 1

# Preset plotting parameters
axs = [None] * len(tags)
nrow = 2
ncol = 3
gs = gridspec.GridSpec(nrow, ncol,
         wspace=0.05, hspace=0.0,
         top=0.2, bottom=0.05,
         left=0.5/(ncol+1), right=1-0.5/(ncol+1))
# Plot
total_feature_att = dict()
for j, dataset in enumerate(datasets):
    for tag in tags:
        ling_feature_att[tag] = np.absolute(torch.load(dataset + '/ling_feature_attentions/' + attention_type + "_" + tag + ".pt").numpy())
        global_max = np.max(ling_feature_att[tag]) if np.max(ling_feature_att[tag]) > global_max else global_max
        global_min = np.min(np.max(ling_feature_att[tag])) if np.min(np.max(ling_feature_att[tag])) < global_min else global_min

    for i, tag in enumerate(tags):
        row = 0 if i < 3 else 1
        col = i if i < 3 else i - 3
        ax = plt.subplot(gs[row, col])
        if attention_type == "effective":
            im = ax.imshow(np.expand_dims(ling_feature_att[tag], axis=0), vmin=global_min, vmax=global_max, cmap='Blues')
        else:
            im = ax.imshow(np.expand_dims(ling_feature_att[tag], axis=0), vmin=global_min, vmax=global_max, cmap='Greens')
        if tag in total_feature_att.keys():
            total_feature_att[tag] += np.expand_dims(ling_feature_att[tag], axis=0)
        else:
            total_feature_att[tag] = np.expand_dims(ling_feature_att[tag], axis=0)

        ax.set_title(tag, fontsize=7, pad=1.0, fontweight='bold')
        plt.xticks(np.arange(0, 12, 1.0))
        ax.xaxis.set_ticks_position('none')
        ax.tick_params(axis='both', which='major', pad=0.1)
        ax.xaxis.labelpad = 0.3
        ax.axes.get_yaxis().set_visible(False)
        ax.label_outer()
        axs[i] = ax
    # plt.suptitle(dataset, fontsize=10, y=0.23, fontweight='bold')
    plt.savefig(attention_type+ "_"+ dataset + "_tags.eps", format='eps')
    plt.savefig(attention_type+ "_"+ dataset + "_tags.png")

# Plot task average
for i, tag in enumerate(tags):
        row = 0 if i < 3 else 1
        col = i if i < 3 else i - 3
        ax = plt.subplot(gs[row, col])
        if attention_type == "effective":
            im = ax.imshow(total_feature_att[tag]/len(datasets), vmin=global_min, vmax=global_max, cmap='Blues')
        else:
            im = ax.imshow(total_feature_att[tag]/len(datasets), vmin=global_min, vmax=global_max, cmap='Greens')

        ax.set_title(tag, fontsize=7, pad=1.0, fontweight='bold')
        plt.xticks(np.arange(0, 12, 1.0))
        ax.xaxis.set_ticks_position('none')
        ax.tick_params(axis='both', which='major', pad=0.1)
        ax.xaxis.labelpad = 0.3
        ax.axes.get_yaxis().set_visible(False)
        ax.label_outer()
        axs[i] = ax
plt.savefig(attention_type+ "_avg_tags.eps", format='eps')
plt.savefig(attention_type+ "_avg_tags.png")
