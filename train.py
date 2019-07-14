from dataloader import videoDataset, transform
from model import Scoring
import torch
import torch.utils.data as data
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
from scipy.stats import spearmanr as sr
import matplotlib.pyplot as plt


train_loss=[]
test_loss=[]
epoch_num=150
def train_shuffle(min_mse=200, max_corr=0):
    trainset = videoDataset(root="figure_skating/c3d_feat",
                            label="data/train_dataset.txt", suffix=".npy", transform=transform, data=None)
    trainLoader = torch.utils.data.DataLoader(trainset,batch_size=128, shuffle=True, num_workers=0)
    testset = videoDataset(root="figure_skating/c3d_feat",
                           label="data/test_dataset.txt", suffix='.npy', transform=transform, data=None)
    testLoader = torch.utils.data.DataLoader(testset,batch_size=64, shuffle=False, num_workers=0)

    # build the model
    scoring = Scoring(feature_size=4096)
    if torch.cuda.is_available():
        scoring.cuda()  # turn the model into gpu
    total_params = sum(p.numel() for p in scoring.parameters() if p.requires_grad)
    optimizer = optim.Adam(params=scoring.parameters(), lr=0.0005)  # Adam
    for epoch in range(epoch_num):
        print("Epoch:  " + str(epoch) + "; Total Params: %d" % total_params)
        total_regr_loss = 0
        total_sample = 0
        for i, (features, scores) in enumerate(trainLoader):  # get mini-batch
            print("%d batches have done" % i)
            if torch.cuda.is_available():
                features = Variable(features).cuda()
                scores = Variable(scores).cuda()
            logits, penal = scoring(features)   # features.shape=(128,300,4096)
            if penal is None:
                regr_loss = scoring.loss(logits, scores)
            else:
                regr_loss = scoring.loss(logits, scores) + penal

            # back propagation
            optimizer.zero_grad()  # PyTorch accumulates the gradients, so clean it every step of backward.
            regr_loss.backward()
            optimizer.step()
            total_regr_loss += regr_loss.data.item() * scores.shape[0]
            total_sample += scores.shape[0]

        loss=total_regr_loss / total_sample
        train_loss.append(loss)
        print("Regression Loss: " + str(loss) + '\n')

        ### the rest is used to evaluate the model with the test dataset ###
        scoring.eval()   # turn the model into evaluation mode(for batch-normalization / dropout layer)
        val_pred = []
        val_sample = 0
        val_loss = 0
        val_truth = []
        for j, (features, scores) in enumerate(testLoader):
            val_truth.append(scores.numpy())
            if torch.cuda.is_available():
                features = Variable(features).cuda()
                scores = Variable(scores).cuda()
            regression, _ = scoring(features)
            val_pred.append(regression.data.cpu().numpy())
            regr_loss = scoring.loss(regression, scores)
            val_loss += (regr_loss.data.item()) * scores.shape[0]
            val_sample += scores.shape[0]
        val_truth = np.concatenate(val_truth)
        val_pred = np.concatenate(val_pred)
        val_sr, _ = sr(val_truth, val_pred)
        if val_loss / val_sample < min_mse:
            torch.save(scoring.state_dict(), 'S_LSTM+M_LSTM+PCS.pt')
        min_mse = min(min_mse, val_loss / val_sample)
        max_corr = max(max_corr, val_sr)
        loss=val_loss / val_sample
        test_loss.append(loss)
        print("Val Loss: {:.2f} Correlation: {:.2f} Min Val Loss: {:.2f} Max Correlation: {:.2f}"
              .format(loss, val_sr, min_mse, max_corr))

        scoring.train()  # turn back to train mode



min_mse = 200
max_corr = 0
train_shuffle(min_mse, max_corr)

# plot
plt.plot(range(0,epoch_num),train_loss,'r-')
plt.plot(range(0,epoch_num),test_loss,'g-')
plt.legend(['train_mse','test_mse'])
plt.title("S_LSTM+M_LSTM: PCS")
plt.show() 
