import os
import torch
import tqdm
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd


def test(data_test, device, model, loss_fn):
    """Test the loss and accuracy of the model on a data set 

    Args:
        data_test: data set
        device: device which the model runs on
        model: training model
        loss_fn: loss function

    Returns:
        accuracy and loss
    """
    test_loss = 0.0
    correct, total = 0, 0
    for x_batch, y_batch, _ in data_test:
        with torch.no_grad():

            x_batch = x_batch.to(device, dtype=torch.float)
            y_batch = y_batch.to(device, dtype=torch.float) 

            model.eval()
            output = model(x_batch)
            correct += torch.sum(torch.argmax(output,axis = 1) == torch.argmax(y_batch, axis = 1))
            total += len(y_batch)

            loss = loss_fn(output.to(dtype=torch.float32), y_batch.to(dtype=torch.float32))
            test_loss += loss.item() * x_batch.size(0)
    acc = correct/total
    return acc.item(), test_loss/total


def save_confusion_des(y_ls, y_pred_ls, des_ls, labels, name, path_des):
    """Save the confusion matrix and des information
    """
    font = {'size': 10}
    plt.rc('font', **font)
    plt.rcParams['figure.figsize'] = [8, 6]
    cm_display =  metrics.ConfusionMatrixDisplay.from_predictions(
        y_ls, 
        y_pred_ls, 
        values_format = '.3f',
        display_labels= labels,
        normalize='true', 
        include_values=True, 
        cmap=plt.cm.Blues, 
        colorbar=False
        )

    plt.title(name) 
    plt.savefig(path_des+'fig/cm-'+name+'.png')
    plt.close()
    df = pd.concat([
        pd.DataFrame(y_ls, columns=['label']), 
        pd.DataFrame(y_pred_ls, columns=['label_pred']), 
        (des_ls.drop(des_ls.columns[[0]],axis=1)).reset_index(drop=True)
        ], 
        axis = 1
        ).reset_index()
    df.to_csv(
        path_des + 'des/des-'+name+'.csv',
        index=False,
        )  


def save_result(set, set_name, device, model, args):
    """Evaluate the model on the data sets, and save the confusion matrix, and
    des information with labels and predicted labels.
    """
    y_ls = []
    y_pred_ls = []
    des_ls = []
    for x_batch, y_batch, des in set:
        x_batch = x_batch.to(device)

        model.eval()
        model.to(device)
        y_batch_pred = model(x_batch)
        y_batch_pred = torch.argmax(y_batch_pred,axis = 1)

        y_batch = torch.argmax(y_batch,axis = 1)
        y_batch_pred = y_batch_pred.cpu().detach().numpy()
        y_ls.append(list(y_batch))
        y_pred_ls.append(list(y_batch_pred))
        des_ls.append(pd.DataFrame.from_dict(des))
    
    y_ls = np.concatenate(y_ls)
    y_pred_ls = np.concatenate(y_pred_ls)
    des_ls = pd.concat(des_ls, axis = 0)

    os.makedirs(os.path.join(args.result.path_des,'des'), exist_ok=True)
    os.makedirs(os.path.join(args.result.path_des,'fig'), exist_ok=True)
    os.makedirs(os.path.join(args.result.path_des,'model'), exist_ok=True)

    save_confusion_des(
        y_ls, 
        y_pred_ls, 
        des_ls, 
        labels=args.result.labels, 
        name=args.result.name+set_name,
        path_des=args.result.path_des
        )

    torch.save(model.state_dict(), args.result.path_des+'model/model_' + args.wandb.project + '_' + args.result.name+set_name + '.pt')
