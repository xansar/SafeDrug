

import matplotlib.pyplot as plt
import numpy as np

plt.rc('font', family='Times New Roman')

import seaborn as sns
import pandas as pd
# sns.set()
def draw_dist_fig(pth):
    dist = np.loadtxt(pth)
    box_dp, box_med, box_mem = dist[:, 0], dist[:, 1], dist[:, 2]
    plt.style.use('seaborn-v0_8-deep')
    plt.figure(figsize=(8, 6), dpi=600)  # 设置画布的尺寸
    # plt.title('Multi Channel ', fontsize=20)  # 标题，并设定字号大小
    labels = ['Health', 'Med', 'Memory']  # 图例
    data = np.concatenate([box_dp, box_med, box_mem])
    label = ['Health'] * dist.shape[0] + ['Med'] * dist.shape[0] + ['Memory'] * dist.shape[0]
    df = pd.DataFrame({'Weight': data, 'Channels': label})
    # facecolors = ['#9CD2B8', '#FFDBB6', '#FFC0B8']
    # colors = ['#6E9482', '#DBBC9D', '#E0A9A2']
    # ax = sns.violinplot(x=df['Channels'], y=df['Weight'], palette=facecolors, scale='count')
    ax = sns.violinplot(x=df['Channels'], y=df['Weight'], palette='Set2', scale='count')
    ax.set_xticklabels(labels, fontsize=20)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=20)
    ax.set_ylabel('Weight', fontsize=24)  # 设置Y坐标轴标签字体
    ax.set_xlabel('Channels', fontsize=24)  # 设置X坐标轴标签字体
    # plt.show()
    plt.savefig('channel_dist.pdf', dpi=600)


    # fig = plt.boxplot([box_dp, box_med, box_mem], labels=labels, vert=True, notch=True, patch_artist=True,
    #                   whiskerprops={'linewidth': 1},
    #                   capprops = {'linewidth': 1},
    #                   flierprops={'markersize': 4, 'alpha': 0.1}
    #                   )  # grid=False：代表不显示背景中的网格线
    # # data.boxplot()#画箱型图的另一种方法，参数较少，而且只接受dataframe，不常用
    # for box, fc, c in zip(fig['boxes'], facecolors, colors):  # 对箱线图设置颜色
    #     box.set(linewidth=1)
    #     box.set(color=c)
    #     box.set(facecolor=fc)
    #
    #     # box.set(markerfacecolor=c)
    # whisker_colors = ['#6E9482', '#6E9482', '#DBBC9D', '#DBBC9D', '#E0A9A2', '#E0A9A2']
    # for whisker, c in zip(fig['whiskers'], whisker_colors):
    #     # print(whisker)
    #     whisker.set(color=c)
    # for cap, c in zip(fig['caps'], whisker_colors):
    #     # print(whisker)
    #     cap.set(color=c)
    # for flier, c in zip(fig['fliers'], colors):
    #     # print(flier)
    #     flier.set(marker='o', color=c, alpha=0.1)
    # plt.show()  # 显示图像

def draw_bar_plot():
    data = [
    {'value': 53.06, 'std': 0.19, 'metric': 'Jaccard', 'Model': 'HypeMed'},
    {'value': 68.44, 'std': 0.17, 'metric': 'F1-score', 'Model': 'HypeMed'},
    {'value': 77.73, 'std': 0.39, 'metric': 'PRAUC', 'Model': 'HypeMed'},
    {'value': 6.51, 'std': 0.05, 'metric': 'ddi_rate', 'Model': 'HypeMed'},
    {'value': 23.44, 'std': 0.13, 'metric': 'avg_med', 'Model': 'HypeMed'},
    {'value': 52.78, 'std': 0.23, 'metric': 'Jaccard', 'Model': 'HypeMed-H'},
    {'value': 68.19, 'std': 0.19, 'metric': 'F1-score', 'Model': 'HypeMed-H'},
    {'value': 77.47, 'std': 0.32, 'metric': 'PRAUC', 'Model': 'HypeMed-H'},
    {'value': 6.24, 'std': 0.06, 'metric': 'ddi_rate', 'Model': 'HypeMed-H'},
    {'value': 23.31, 'std': 0.12, 'metric': 'avg_med', 'Model': 'HypeMed-H'},
    {'value': 52.21, 'std': 0.22, 'metric': 'Jaccard', 'Model': 'HypeMed-P'},
    {'value': 67.69, 'std': 0.21, 'metric': 'F1-score', 'Model': 'HypeMed-P'},
    {'value': 77.21, 'std': 0.32, 'metric': 'PRAUC', 'Model': 'HypeMed-P'},
    {'value': 6.29, 'std': 0.04, 'metric': 'ddi_rate', 'Model': 'HypeMed-P'},
    {'value': 23.43, 'std': 0.13, 'metric': 'avg_med', 'Model': 'HypeMed-P'},
    {'value': 52.58, 'std': 0.15, 'metric': 'Jaccard', 'Model': 'HypeMed-M'},
    {'value': 68.00, 'std': 0.14, 'metric': 'F1-score', 'Model': 'HypeMed-M'},
    {'value': 77.55, 'std': 0.34, 'metric': 'PRAUC', 'Model': 'HypeMed-M'},
    {'value': 6.26, 'std': 0.05, 'metric': 'ddi_rate', 'Model': 'HypeMed-M'},
    {'value': 23.56, 'std': 0.12, 'metric': 'avg_med', 'Model': 'HypeMed-M'},
]


    df = pd.DataFrame(data, columns=['value', 'std', 'Model', 'metric'], dtype=str)
    df[['value', 'std']] = df[['value', 'std']].apply(pd.to_numeric)
    # print(df)
    # plt.figure(figsize=(8, 6), dpi=100)  # 设置画布的尺寸
    fig, axes = plt.subplots(3, 1, figsize=(10, 5.5), dpi=600)
    metrics_name = ['Jaccard', 'F1-score', 'PRAUC', 'ddi_rate', 'avg_med']
    x_lims = [
        (51.8, 53.3),
        (67.4, 68.7),
        (76.8, 78.25),
        (6.15, 6.6),
        (23.1, 24),
    ]
    for i in range(3):
        fig = sns.barplot(data=df[df.metric==metrics_name[i]], y="Model", x="value", palette="Set2", ax=axes[i])
        axes[i].errorbar(x=df[df.metric==metrics_name[i]]['value'],
                     y=df[df.metric==metrics_name[i]]['Model'],
                     xerr=df[df.metric==metrics_name[i]]['std'],
                     ls='none', capsize=3,
                     ecolor='black'
                     )
        axes[i].set_ylabel('', fontsize=0)  # 设置Y坐标轴标签字体
        axes[i].set_yticklabels(['HypeMed', 'HypeMed-H', 'HypeMed-P', 'HypeMed-M'], fontsize=20)
        plt.setp(axes[i].get_xticklabels(), fontsize=20)
        # axes[i].set_xticklabels(axes[i].get_xticklabels(which='both'), fontsize=16)
        axes[i].set_xlabel(metrics_name[i], fontsize=24)  # 设置X坐标轴标签字体
        # fig.set_xlim(0, 20)  # 限制x的值为[0,20]
        fig.set_xlim(*(x_lims[i]))  # 限制y的值为[0,1]
    # plt.legend()
    # plt.subplots_adjust(hspace=0.6)
    plt.tight_layout()
    # plt.show()
    plt.savefig('channel_result.pdf', dpi=600)


if __name__ == '__main__':
    # draw_dist_fig('total_gates_lst.txt')
    draw_bar_plot()