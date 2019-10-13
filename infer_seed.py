# 这里写生成seed的代码
import torch
import torchvision
import argparse
import numpy as np
import networks.vgg16_gcn
from utils import pyutils
import dataset.data
import os
import imageio
import tqdm
from utils import imgutils
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.backends import cudnn
cudnn.enabled = True


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="vgg16_gcn0.pth", type=str)
    parser.add_argument("--network", default="networks.vgg16_gcn", type=str)
    parser.add_argument("--val_list", default="dataset/val.txt", type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--dataset_root", default="dataset/VOCdevkit/VOC2012", type=str)
    parser.add_argument("--out_cam_pred", default="cam_pred", type=str)

    args = parser.parse_args()
    model = networks.vgg16_gcn.Net()
    model = networks.vgg16_gcn.GCNVggNet(model, num_classes=20, t=0.4, adj_file='files/voc_adj.pkl')   # 这里有问题 怎么能加载出整体的模型
    model.load_state_dict(torch.load(args.weights))

    model.eval()
    model.cuda()

    val_dataset = dataset.data.VOC12ClsDatasetMSF(set=args.val_list, path=args.dataset_root,
                                               inp_name='files/voc_glove_word2vec.pkl',
                                               scales=(1, 0.5, 1.5, 2.0),
                                               inter_transform=torchvision.transforms.Compose([
                                                   np.asarray,
                                                   model.normalize,
                                                   imgutils.HWC_to_CHW
                                               ]))
    val_data_loader = DataLoader(val_dataset, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    n_gpus = torch.cuda.device_count()
    model_replicas = torch.nn.parallel.replicate(model, list(range(n_gpus)))
    # val_data_loader = tqdm(val_data_loader, desc='Generate seeds')
    for iter, (name, images, target, inp) in enumerate(val_data_loader):
        image_name = name[0]
        label = target[0]
        #print(images)
        image_path = os.path.join(args.dataset_root, 'JPEGImages', image_name +'.jpg')
        orig_img = np.asarray(Image.open(image_path))
        orig_image_size = orig_img.shape[:2]
        #print(orig_image_size)

        def _work(i, img):
            with torch.no_grad():
                with torch.cuda.device(i%n_gpus):
                    #print(img.shape)
                    cam = model_replicas[i%n_gpus].forward_cam(img.cuda(), inp)   # (batch, 14, 14, 20)
                    cam = F.upsample(cam, orig_image_size, mode='bilinear', align_corners=False)[0]  # 直接就上采样了？？？
                    cam = cam.cpu().numpy() * label.clone().view(20, 1, 1).numpy()
                    if i % 2 == 1:
                        cam = np.flip(cam, axis=-1)
                    return cam

        thread_pool = pyutils.BatchThreader(_work, list(enumerate(images)),
                                            batch_size=12, prefetch_size=0, processes=args.num_workers)

        cam_list = thread_pool.pop_results()

        sum_cam = np.sum(cam_list, axis=0)
        norm_cam = sum_cam / (np.max(sum_cam, (1, 2), keepdims=True) + 1e-5)

        cam_dict = {}
        for i in range(20):
            if label[i] > 1e-5:
                cam_dict[i] = norm_cam[i]

        if args.out_cam_pred is not None:
            bg_score = [np.ones_like(norm_cam[0])*0.2]  # 默认每个像素属于背景类的概率为0.2？
            pred = np.argmax(np.concatenate((bg_score, norm_cam)), 0)   # 对每个像素输出，看shape,怎么得到的每个点属于的类别
            # scipy.misc.imsave(os.path.join(args.out_cam_pred, img_name + '.png'), pred.astype(np.uint8))
            imageio.imwrite(os.path.join(args.out_cam_pred, image_name + '.png'), pred.astype(np.uint8))

    print("Finish!")
