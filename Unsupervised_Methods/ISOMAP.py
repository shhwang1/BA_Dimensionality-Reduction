import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn import manifold
from sklearn.preprocessing import MinMaxScaler

def isomap(args):

    data = pd.read_csv(args.data_path + args.data_type)

    X_data = data.iloc[:, :-1]
    y_data = data.iloc[:, -1]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_data)

    isomap = manifold.Isomap(n_components = 2)
    new_dim = isomap.fit_transform(X_scaled)

    df = pd.DataFrame(new_dim, columns=['X', 'Y'])
    df['label'] = y_data

    fig = plt.figure()
    fig.suptitle('Isomap', fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111)

    for i in np.unique(y_data):
        plt.scatter(df[df.label == i].X, df[df.label == i].Y, label=i)

    plt.legend(bbox_to_anchor=(1.25, 1))
    plt.show()