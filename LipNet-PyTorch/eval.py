import os
import time

import numpy as np
import torch
import torch.nn as nn
from dataset import MyDataset
from model import LipNet
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

if (__name__ == '__main__'):
    opt = __import__('options')
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    writer = SummaryWriter()


def dataset2dataloader(dataset, num_workers=opt.num_workers, shuffle=True):
    return DataLoader(dataset,
                      batch_size=opt.batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      drop_last=False)


def ctc_decode(y):
    y = y.argmax(-1)
    return [MyDataset.ctc_arr2txt(y[_], start=1) for _ in range(y.size(0))]


def eval(model, net):
    with torch.no_grad():
        dataset = MyDataset(opt.video_path,
                            opt.anno_path,
                            opt.val_list,
                            opt.vid_padding,
                            opt.txt_padding,
                            'test')

        print('num_test_data:{}'.format(len(dataset.data)))
        model.eval()
        loader = dataset2dataloader(dataset, shuffle=False)

        wer = []
        cer = []
        tic = time.time()
        for (i_iter, input) in enumerate(loader):
            vid = input.get('vid').cuda()
            txt = input.get('txt').cuda()

            y = net(vid)

            pred_txt = ctc_decode(y)

            truth_txt = [MyDataset.arr2txt(txt[_], start=1) for _ in range(txt.size(0))]
            wer.extend(MyDataset.wer(pred_txt, truth_txt))
            cer.extend(MyDataset.cer(pred_txt, truth_txt))

            if (i_iter % opt.display == 0):

                print(''.join(101 * '-'))
                print('{:<50}|{:>50}'.format('predict', 'truth'))
                print(''.join(101 * '-'))
                for (predict, truth) in list(zip(pred_txt, truth_txt))[:10]:
                    print('{:<50}|{:>50}'.format(predict, truth))
                print(''.join(101 * '-'))
                print('test_iter={},wer={},cer={}'.format(i_iter, np.array(wer).mean(), np.array(cer).mean()))
                print(''.join(101 * '-'))

            writer.add_scalar('wer', np.array(wer).mean(), i_iter)
            writer.add_scalar('cer', np.array(cer).mean(), i_iter)

        return np.array(wer).mean(), np.array(cer).mean()


if __name__ == '__main__':
    print("Loading options...")
    # load model
    # model = LipNet(isTransformer=True, isDense=True)
    model = LipNet()
    model = model.cuda()
    net = nn.DataParallel(model).cuda()

    model_dict = model.state_dict()
    # Load the weight files
    pretrained_dict = torch.load(opt.weights)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                       k in model_dict.keys() and v.size() == model_dict[k].size()}

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    print("Loaded Model")
    print("Evaluating on test set")
    (wer, cer) = eval(model, net)

    print('Final Metrics, wer={}, cer={}'.format(wer, cer))