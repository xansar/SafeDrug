
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

plt.rc('font', family='Times New Roman')

from  matplotlib.colors import  rgb2hex
import seaborn as sns
# sns.set()
def pca(x, dim):
    model = PCA(n_components=dim)
    low_x = model.fit_transform(x)
    return low_x


def tsne(x, dim):
    # t-SNE的降维与可视化
    assert dim < 4
    ts = TSNE(n_components=dim, init='pca', random_state=0)
    # 训练模型
    y = ts.fit_transform(x)
    return y

def visualization(X, n_clusters=8):
    X = X.detach().cpu().numpy()

    # 先降维
    X = pca(X, 16)
    # 再做可视化
    visual_X = tsne(X, 2)

    # model = KMeans(n_clusters=n_clusters)
    # model.fit(visual_X)
    # label_pred = model.labels_  # 获取聚类标签

    # model = GaussianMixture(n_components=n_clusters, n_init=10, random_state=42)
    # label_pred = model.fit_predict(visual_X)
    label_pred = None
    return visual_X, label_pred

    # plt.show()

def draw_fig(ax, visual_X, label_pred, title, n_clusters, seed):
    np.random.seed(seed)
    colors = tuple([(np.random.random(), np.random.random(), np.random.random()) for i in range(n_clusters)])
    colors = [rgb2hex(x) for x in colors]  # from  matplotlib.colors import  rgb2hex


    # 绘制k-means结果
    sns.scatterplot(x=visual_X[:, 0], y=visual_X[:, 1], hue=label_pred, ax=ax, size=6, linewidth=0, legend=False, palette="tab20")
    # for i in range(n_clusters):
    #     x = visual_X[label_pred == i]
    #     sns.scatterplot(x=x[:, 0], y=x[:, 1], size=4, ax=ax, legend=False, palette="Pastel1")
        # plt.scatter(x[:, 0], x[:, 1], c=colors[i], s=4)
    # plt.legend(loc=2)
    print(f'finish {title}')
    ax.set_title(title, fontsize=24)

def experiment_visual(pth3, pth4):
    # res3 = torch.load(pth3)
    # # res = torch.load('tmp.pkl')
    # X_hat3 = res3['X']
    # E_mem = res3['E']
    # res4 = torch.load(pth4)
    # # res = torch.load('tmp.pkl')
    # X_hat4 = res4['X']
    # E_mem = res4['E']
    # diag3 = visualization(X_hat3['diag'], 30)
    # proc3 = visualization(X_hat3['proc'], 30)
    # med3 = visualization(X_hat3['med'], 30)
    # diag4 = visualization(X_hat4['diag'], 30)
    # proc4 = visualization(X_hat4['proc'], 30)
    # med4 = visualization(X_hat4['med'], 30)
    #
    # clusters = {
    #     'diag3': diag3,
    #     'proc3': proc3,
    #     'med3': med3,
    #     'diag4': diag4,
    #     'proc4': proc4,
    #     'med4': med4,
    # }
    # torch.save(clusters, 'embed_cluster.pkl')
    print('begin fig')
    seed = 548
    clusters = torch.load('embed_cluster.pkl')
    # print(plt.style.available)
    # plt.style.use('seaborn-v0_8-deep')
    plt.figure(figsize=(12, 8), dpi=600)
    ax1 = plt.subplot(231)
    draw_fig(ax1, *clusters['diag3'], 'Diagnosis Nodes (III)', 30, seed)
    ax3 = plt.subplot(232)
    draw_fig(ax3, *clusters['proc3'], 'Procedure Nodes (III)', 30, seed)
    # visualization(X_hat3['proc'], ax3, 'Procedure Nodes (III)', 24)

    ax2 = plt.subplot(233)
    draw_fig(ax2, *clusters['med3'], 'Medication Nodes (III)', 30, seed)
    # visualization(X_hat3['med'], ax2, 'Medication Nodes (III)', 20)

    ax4 = plt.subplot(234)
    draw_fig(ax4, *clusters['diag4'], 'Diagnosis Nodes (IV)', 30, seed)
    # visualization(X_hat4['diag'], ax4, 'Diagnosis Nodes (IV)', 30)
    ax6 = plt.subplot(235)
    draw_fig(ax6, *clusters['proc4'], 'Procedure Nodes (IV)', 30, seed)
    # visualization(X_hat4['proc'], ax3, 'Procedure Nodes (IV)', 24)
    ax5 = plt.subplot(236)
    draw_fig(ax5, *clusters['med4'], 'Medication Nodes (IV)', 30, seed)
    # visualization(X_hat4['med'], ax2, 'Medication Nodes (IV)', 20)

    # plt.subplots_adjust()
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'nodes_34.pdf', dpi=600)
    #
    # visualization(E_mem['diag'], 'Diagnosis Edges', 24)
    # visualization(E_mem['proc'], 'Procedure Edges', 24)
    # visualization(E_mem['med'], 'Medication Edges', 24)


if __name__ == '__main__':
    experiment_visual('save_embed_mimic3.pkl', 'embed_mimic_4.pkl')
    # res3 = torch.load('tmp3.pkl')
    # # res = torch.load('tmp.pkl')
    # X_hat3 = res3['X']
    # E_mem = res3['E']
    # for n in ['diag', 'proc', 'med']:
    #     visual_X, label_pred = visualization(X_hat3[n], 30)
    #     plt.scatter(visual_X[:, 0], visual_X[:, 1])
    #     plt.show()
    #     print(n)
