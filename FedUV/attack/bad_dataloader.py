from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset, Subset
import torch
import numpy as np
from tqdm import tqdm
import time
from PIL import Image

class DatasetBD(Dataset):
    def __init__(self, opt, full_dataset, inject_portion, transform=None, mode="train", device=torch.device("cuda"), distance=20):
        self.dataset = self.addTrigger(full_dataset, opt.target_label, inject_portion, mode, distance, opt.trig_w, opt.trig_h, opt.trigger_type, opt.target_type)
        self.device = device
        self.transform = transform

    def __getitem__(self, item):
        img = self.dataset[item][0]
        label = self.dataset[item][1]
        
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img.astype(np.uint8))
        
        if self.transform:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.dataset)

    def addTrigger(self, dataset, target_label, inject_portion, mode, distance, trig_w, trig_h, trigger_type, target_type):
        print("Generating " + mode + "bad Imgs")
        portion = min(inject_portion * len(target_label), 1.0)
        all_perm = np.random.permutation(len(dataset))[0: int(len(dataset) * portion)]
        # dataset
        dataset_ = list()

        perms = np.array_split(all_perm, len(target_label))
        for idx, perm in enumerate(perms):
            cnt = 0
            for i in tqdm(range(len(dataset))):
                data = dataset[i]

                if target_type == 'all2one':

                    if mode == 'train':
                        img = np.array(data[0])
                        width = img.shape[0]
                        height = img.shape[1]
                        if i in perm:
                            # select trigger
                            img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type[idx])

                            # change target
                            dataset_.append((img, target_label[idx]))
                            cnt += 1
                        else:
                            dataset_.append((img, data[1]))

                    else:
                        if data[1] == target_label[idx]:
                            continue

                        img = np.array(data[0])
                        width = img.shape[0]
                        height = img.shape[1]
                        if i in perm:
                            img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type[idx])

                            dataset_.append((img, target_label[idx]))
                            cnt += 1
                        else:
                            dataset_.append((img, data[1]))

                # all2all attack
                elif target_type == 'all2all':

                    if mode == 'train':
                        img = np.array(data[0])
                        width = img.shape[0]
                        height = img.shape[1]
                        if i in perm:

                            img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)
                            target_ = self._change_label_next(data[1])

                            dataset_.append((img, target_))
                            cnt += 1
                        else:
                            dataset_.append((img, data[1]))

                    else:

                        img = np.array(data[0])
                        width = img.shape[0]
                        height = img.shape[1]
                        if i in perm:
                            img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)

                            target_ = self._change_label_next(data[1])
                            dataset_.append((img, target_))
                            cnt += 1
                        else:
                            dataset_.append((img, data[1]))

                # clean label attack
                elif target_type == 'cleanLabel':

                    if mode == 'train':
                        img = np.array(data[0])
                        width = img.shape[0]
                        height = img.shape[1]

                        if i in perm:
                            if data[1] == target_label[idx]:

                                img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)

                                dataset_.append((img, data[1]))
                                cnt += 1

                            else:
                                dataset_.append((img, data[1]))
                        else:
                            dataset_.append((img, data[1]))

                    else:
                        if data[1] == target_label:
                            continue

                        img = np.array(data[0])
                        width = img.shape[0]
                        height = img.shape[1]
                        if i in perm:
                            img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type[idx])

                            dataset_.append((img, target_label[idx]))
                            cnt += 1
                        else:
                            dataset_.append((img, data[1]))

        time.sleep(0.01)
        print("Injecting Over: " + str(cnt) + "Bad Imgs, " + str(len(dataset) - cnt) + "Clean Imgs")


        return dataset_


    def _change_label_next(self, label):
        label_new = ((label + 1) % 10)
        return label_new

    def selectTrigger(self, img, width, height, distance, trig_w, trig_h, triggerType):

        # print("triggerTyep: ", triggerType)
        assert triggerType in ['squareTrigger', 'gridTrigger', 'fourCornerTrigger', 'randomPixelTrigger',
                            'signalTrigger', 'trojanTrigger', 'blendTrigger']

        if triggerType == 'squareTrigger':
            img = self._squareTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'gridTrigger':
            img = self._gridTriger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'fourCornerTrigger':
            img = self._fourCornerTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'randomPixelTrigger':
            img = self._randomPixelTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'signalTrigger':
            img = self._signalTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'trojanTrigger':
            img = self._trojanTrigger(img, width, height, distance, trig_w, trig_h)
        
        elif triggerType == 'blendTrigger':
            img = self._blendTrigger(img, width, height, distance, trig_w, trig_h)

        else:
            raise NotImplementedError

        return img

    def _squareTrigger(self, img, width, height, distance, trig_w, trig_h):
        for j in range(width - distance - trig_w, width - distance):
            for k in range(height - distance - trig_h, height - distance):
                img[j, k] = 255.0

        return img

    def _gridTriger(self, img, width, height, distance, trig_w, trig_h):

        img[width - 1][height - 1] = 255
        img[width - 1][height - 2] = 0
        img[width - 1][height - 3] = 255

        img[width - 2][height - 1] = 0
        img[width - 2][height - 2] = 255
        img[width - 2][height - 3] = 0

        img[width - 3][height - 1] = 255
        img[width - 3][height - 2] = 0
        img[width - 3][height - 3] = 0

        # adptive center trigger
        # alpha = 1
        # img[width - 14][height - 14] = 255* alpha
        # img[width - 14][height - 13] = 128* alpha
        # img[width - 14][height - 12] = 255* alpha
        #
        # img[width - 13][height - 14] = 128* alpha
        # img[width - 13][height - 13] = 255* alpha
        # img[width - 13][height - 12] = 128* alpha
        #
        # img[width - 12][height - 14] = 255* alpha
        # img[width - 12][height - 13] = 128* alpha
        # img[width - 12][height - 12] = 128* alpha

        return img

    def _fourCornerTrigger(self, img, width, height, distance, trig_w, trig_h):
        # right bottom
        img[width - 1][height - 1] = 255
        img[width - 1][height - 2] = 0
        img[width - 1][height - 3] = 255

        img[width - 2][height - 1] = 0
        img[width - 2][height - 2] = 255
        img[width - 2][height - 3] = 0

        img[width - 3][height - 1] = 255
        img[width - 3][height - 2] = 0
        img[width - 3][height - 3] = 0

        # left top
        img[1][1] = 255
        img[1][2] = 0
        img[1][3] = 255

        img[2][1] = 0
        img[2][2] = 255
        img[2][3] = 0

        img[3][1] = 255
        img[3][2] = 0
        img[3][3] = 0

        # right top
        img[width - 1][1] = 255
        img[width - 1][2] = 0
        img[width - 1][3] = 255

        img[width - 2][1] = 0
        img[width - 2][2] = 255
        img[width - 2][3] = 0

        img[width - 3][1] = 255
        img[width - 3][2] = 0
        img[width - 3][3] = 0

        # left bottom
        img[1][height - 1] = 255
        img[2][height - 1] = 0
        img[3][height - 1] = 255

        img[1][height - 2] = 0
        img[2][height - 2] = 255
        img[3][height - 2] = 0

        img[1][height - 3] = 255
        img[2][height - 3] = 0
        img[3][height - 3] = 0

        return img

    def _randomPixelTrigger(self, img, width, height, distance, trig_w, trig_h):
        alpha = 0.2
        mask = np.random.randint(low=0, high=256, size=(width, height), dtype=np.uint8)
        blend_img = (1 - alpha) * img + alpha * mask.reshape((width, height, 1))
        blend_img = np.clip(blend_img.astype('uint8'), 0, 255)

        # print(blend_img.dtype)
        return blend_img

    def _signalTrigger(self, img, width, height, distance, trig_w, trig_h):
        alpha = 0.2
        # load signal mask
        signal_mask = np.load('trigger/signal_cifar10_mask.npy')
        blend_img = (1 - alpha) * img + alpha * signal_mask.reshape((width, height, 1))  # FOR CIFAR10
        blend_img = np.clip(blend_img.astype('uint8'), 0, 255)

        return blend_img

    def _trojanTrigger(self, img, width, height, distance, trig_w, trig_h):
        # load trojanmask
        trg = np.load('trigger/best_square_trigger_cifar10.npz')['x']
        # trg.shape: (3, 32, 32)
        trg = np.transpose(trg, (1, 2, 0))
        img_ = np.clip((img + trg).astype('uint8'), 0, 255)

        return img_
    
    def _blendTrigger(self, img, width, height, distance, trig_w, trig_h):
        import cv2
        alpha = 0.2
        # 读取 trigger
        trigger = cv2.imread('trigger/hello_kitty.png')
        trigger = cv2.cvtColor(trigger, cv2.COLOR_BGR2RGB)

        # resize 到和输入图像一样
        trigger = cv2.resize(trigger, (img.shape[1], img.shape[0]))

        # 转 float
        img_f = img.astype(np.float32)
        trigger_f = trigger.astype(np.float32)

        # blend
        poison_img = (1 - alpha) * img_f + alpha * trigger_f
        # clip
        poison_img = np.clip(poison_img, 0, 255).astype(np.uint8)
        return poison_img

        # self.trigger = Image.open('triggers/blended.jpg').resize((self.img_size, self.img_size), Image.BILINEAR)
        # img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        # _img = Image.blend(img, self.trigger, 0.1)
        # return _img


def get_bad_dataloader(ds_name, train_bs, dataidxs=None, args=None, prop=0., client_id=None):
    if ds_name == 'cifar10':
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                        std=[0.247, 0.243, 0.261])
        transform_train = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            #transforms.ColorJitter(brightness=noise_level),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize,
        ])
        # data prep for test set
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            ])
        
        # test
        if prop == 1.:
            dataset = CIFAR10(root=args.datadir, train=False, download=False)
            bad_ds = DatasetBD(args, dataset, 
                            inject_portion=1.,
                            transform=transform_test,
                            mode="test",
                            device=args.device,
                            )
        # train
        else:
            dataset = CIFAR10(root=args.datadir, train=True, download=False)
            local_dataset = Subset(dataset, dataidxs)

            import copy
            settings = copy.deepcopy(args)
            if client_id is not None:
                settings.target_label, settings.trigger_type = [], []
                i = client_id % len(args.trigger_type)
                settings.target_label.append(args.target_label[i])
                settings.trigger_type.append(args.trigger_type[i])
            bad_ds = DatasetBD(settings, 
                            local_dataset, 
                            inject_portion=prop,
                            transform=transform_train,
                            mode="train",
                            device=args.device,
                            )
            # bad_ds = DatasetSplit(all_bad_ds, dataidxs)
        
        bad_dl = torch.utils.data.DataLoader(dataset=bad_ds, batch_size=train_bs, num_workers=8, drop_last=True, shuffle=True, pin_memory=True, persistent_workers =True,)

    else:
        raise Exception(f"The dataset {ds_name} has not been supported!")

    return bad_dl