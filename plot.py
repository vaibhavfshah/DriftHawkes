import pandas as pd
import matplotlib.pyplot as plt

def plot_losses(dataframe, path):
    df = dataframe
    loss_columns = [col for col in df.columns if col.startswith('Loss_domain_')]

    for col in loss_columns:
        plt.plot(df.index, df[col], label=col)

    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')

    plt.savefig(f'{path}.png')
    # plt.show()
