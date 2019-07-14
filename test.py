from dataloader import videoDataset, transform
from model import Scoring
import torch
import torch.utils.data as data
from torch.autograd import Variable


def Test_shuffle():
    testset = videoDataset(root="figure_skating/c3d_feat",
                   label="data/test_dataset.txt", suffix='.npy', transform=transform, data=None)
    testLoader = torch.utils.data.DataLoader(testset,
                                      batch_size=64, shuffle=False, num_workers=0)

    #build the model
    scoring = Scoring(feature_size=4096)
    if torch.cuda.is_available():
        scoring.cuda()  #turn the model into gpu
    scoring.load_state_dict(torch.load("S_LSTM+M_LSTM+PCS.pt"))
    scoring.eval()

    # val_pred = []
    val_sample = 0
    val_loss = 0
    # val_truth = []
    for j, (features, scores) in enumerate(testLoader):
        # val_truth.append(scores.numpy())
        if torch.cuda.is_available():
            features = Variable(features).cuda()
            scores = Variable(scores).cuda()
        regression, _ = scoring(features)
        # val_pred.append(regression.data.cpu().numpy())
        regr_loss = scoring.loss(regression, scores)
        val_loss += (regr_loss.data.item()) * scores.shape[0]
        val_sample += scores.shape[0]
    print("Val Loss: %.2f"% (val_loss / val_sample))

Test_shuffle()