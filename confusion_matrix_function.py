import time, torch, os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import pandas as pd
import seaborn as sn

def plot_cm(cm, classes, save_file, title, style='simple'):

    if style == 'simple':
        fig = plt.figure(figsize=(12,10))
        ax = fig.subplots(1,1)
        ax.set_title('Confusion Matrix of\n'+title)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        disp = disp.plot(cmap=plt.cm.Blues, ax=ax)

    if style== 'with_axis':
        df_cm = pd.DataFrame(cm)
        df_cm.index.name = 'Actual'
        df_cm.columns.name = 'Predicted'
        fig = plt.figure(figsize=(12,10))
        ax = fig.subplots(1,1)
        ax.set_title('Confusion Matrix of\n'+title)
        res = sn.heatmap(df_cm, annot=True, square=True, cmap='Blues',
                         xticklabels = classes, yticklabels=classes, fmt='g', 
                         ax=ax, cbar_kws={'label': 'Number of Images'})
        res.axhline(y = 0, color = 'k', linewidth = 1)
        res.axhline(y = 16.98, color = 'k', linewidth = 1)
        res.axvline(x = 0, color = 'k', linewidth = 1)
        res.axvline(x = 16.98, color = 'k', linewidth = 1)
    
    if save_file: plt.savefig(save_file, dpi=300)
    plt.show()
    
    
def confusion_matrix(model, dataloader, process_function, classes, device, n_batches=1):
    model = model.to(device)
    model.eval()
    
    cm = torch.zeros(len(classes), len(classes))
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            inputs, labels = process_function(batch)
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for t, p in zip(labels.view(-1), preds.view(-1)):
                cm[t.long(), p.long()] += 1
            if n_batches == i + 1:
                break
    cm = np.array(cm)
    
    print('Sum for true labels:')
    true_counts = np.expand_dims(np.sum(cm, axis=1), 0)
    display(pd.DataFrame(true_counts, columns=classes))

    wrong, right = 0, 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i == j: right+=cm[i,j]
            if i != j: wrong+=cm[i,j]
    print('Accuracy for these batches:', right/(right+wrong))
    return cm.astype(np.int)