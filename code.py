import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
torch.backends.cudnn.benchmark = True
import albumentations as A
from torch.utils.data import Dataset, DataLoader  
import os 
from PIL import Image 
from albumentations.pytorch import ToTensorV2
from torchvision.utils import save_image
import tqdm
import cv2 
import numpy as np 

import os 
import shutil


t1_list = os.listdir("../input/ixi-t1/image slice-T1")
t2_list  = os.listdir("../input/ixit2-slices/image slice-T2")

print(len(t1_list))
print(len(t2_list))

t1_sampled = []
t2_sampled = []

for file1 in t1_list:
    found = False 
    for file2 in t2_list:
        if file1.split("-")[0] == file2.split("-")[0]:
            found = True 
            match_folder = file2 

    if found:  
        #print("matched")
        t1_sampled.append(file1)
        t2_sampled.append(file2)
        #print(file1 +"\t"+ match_folder)

    else: 
        #shutil.rmtree("./image slice-T2/"+file1)
        print("no matched")
        
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "../input"
VAL_DIR = "../input"
LEARNING_RATE = 2e-4
BATCH_SIZE = 16
NUM_WORKERS = 0
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 100
LAMBDA_GP = 10
NUM_EPOCHS = 2
LOAD_MODEL = False
SAVE_MODEL = False
CHECKPOINT_DISC = "disc.pth.tar"
CHECKPOINT_GEN = "gen.pth.tar"

both_transform = A.Compose(
    [A.Resize(width=256, height=256),], additional_targets={"image0": "image"},
)


transform_only_input = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(p=0.2),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

transform_only_mask = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ])

both_transform = A.Compose(
    [A.Resize(width=256, height=256),], additional_targets={"image0": "image"},
)


transform_only_input = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(p=0.2),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

transform_only_mask = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ])

class NORMS(nn.Module):
    def __init__(self, filters, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.conv = nn.Conv2d(filters,filters, 3,1,1, padding_mode="reflect")
        self.conv_gamma = nn.Conv2d(filters,filters, 3,1,1, padding_mode="reflect")
        self.conv_beta = nn.Conv2d(filters,filters, 3,1,1, padding_mode="reflect")

    def forward(self, input_tensor):
        mask = input_tensor 
        x = self.conv(mask)
        gamma = self.conv_gamma(x)
        beta = self.conv_beta(x)
        var = torch.var(input_tensor, dim=(0, 2, 3), keepdim=True)
        mean = torch.mean(input_tensor, dim=(0, 2, 3), keepdim=True)
        std = torch.sqrt(var + self.epsilon)
        normalized = (input_tensor - mean) / std
        output = gamma * normalized + beta
        return output
class Discriminator(nn.Module):
    def __init__(self, in_channels = 3, features = [64, 128, 256, 512]):
        super(Discriminator, self).__init__()
        
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels*2, 
            features[0],
            kernel_size = 4, 
            stride = 2, 
            padding = 1, 
            padding_mode = 'reflect')
        )
        layers = []
        
        in_channel = features[0]
        for out_channel in features[1:]:
            layer = self.block(in_channel, out_channel,stride = 1 if out_channel == features[-1] else 2 )
            in_channel = out_channel 
            layers.append(layer)
            
        layers.append(
            nn.Conv2d(
                in_channel, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"
            ),
        )

            
        self.model = nn.Sequential(*layers)
            
            
        
    def block(self, in_channel, out_channel, stride = 1):
        block_layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 4, stride,1, bias = False, padding_mode = 'reflect'),
            #nn.BatchNorm2d(out_channel),
            NORMS(out_channel),
            nn.LeakyReLU(0.2)
        )
        return block_layer 
    def forward(self, x,y):
        x = torch.cat([x, y] , dim = 1)
        x = self.initial(x)
        x = self.model(x)
        return x 
a = torch.ones([1,3,256,256])
b = torch.ones([1,3, 256, 256])

Discriminator()(a,b).shape

class Block(nn.Module):
    def __init__(self, in_channels,out_channels, down = True, act = 'relu', use_dropout = False):
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias = False, padding_mode = 'reflect') 
            if down else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias = False),
#             nn.BatchNorm2d(out_channels),
            NORMS(out_channels),
            nn.ReLU() if act == 'relu' else nn.LeakyReLU(0.2),
        )
        
        self.dropout = nn.Dropout(0.5)
        self.use_dropout = use_dropout 
        self.down = down 
        
    def forward(self, x): 
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x 
      
class Generator(nn.Module):
    def __init__(self,in_channels=3, features = 64):
        super().__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode = 'reflect'),
            nn.LeakyReLU(0.2)
        ) # channel 64, 
        
        self.down1 = Block(features, features*2, down = True, act = 'leaky', use_dropout = False) # channel = 128
        self.down2 = Block(features*2, features*4, down = True, act = 'leaky', use_dropout = False) # channel = 256
        self.down3 = Block(features*4, features*8, down = True, act = 'leaky', use_dropout = False) # channel = 512
        self.down4 = Block(features*8, features*8, down = True, act = 'leaky', use_dropout = False) # channel = 512
        self.down5 = Block(features*8, features*8, down = True, act = 'leaky', use_dropout = False) # channel = 512
        self.down6 = Block(features*8, features*8, down = True, act = 'leaky', use_dropout = False) # channel = 512 
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*8, features*8, 4, 2,1), nn.ReLU()
        )                                                                    # channel 512 
        
        self.up1 = Block(features*8, features*8, down = False, act = 'relu', use_dropout = True) # channel = 512
        self.up2 = Block(features*8*2, features*8, down = False, act = 'relu', use_dropout = True) # channel = 512
        self.up3 = Block(features*8*2, features*8, down = False, act = 'relu', use_dropout = True) # channel = 512 
        self.up4 = Block(features*8*2, features*8, down = False, act = 'relu', use_dropout = True) # channel 512 
        self.up5 = Block(features*8*2, features*4, down = False, act = 'relu', use_dropout = True) # channel = 256 
        self.up6 = Block(features*4*2, features*2, down = False, act = 'relu', use_dropout = True) # channel = 128
        self.up7 = Block(features*2*2, features, down = False, act = 'relu', use_dropout = True) # channel = 64
        
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features*2, in_channels, kernel_size = 4, stride = 2, padding = 1), 
            nn.Tanh(),
        )
        
        self.d7_conv = nn.Conv2d(512, 512, 3, 1, 1, bias = False, padding_mode = 'reflect')
        self.d7_conv1 = nn.Conv2d(512, 512, 3, 1, 1, bias = False, padding_mode = 'reflect')
        
        self.d6_conv = nn.Conv2d(512, 512, 3, 1, 1, bias = False, padding_mode = 'reflect')
        self.d6_conv1 = nn.Conv2d(512, 512, 3, 1, 1, bias = False, padding_mode = 'reflect')
        
        self.d5_conv = nn.Conv2d(512, 512, 3, 1, 1, bias = False, padding_mode = 'reflect')
        self.d5_conv1 = nn.Conv2d(512, 512, 3, 1, 1, bias = False, padding_mode = 'reflect')
        
        self.d4_conv = nn.Conv2d(512, 512, 3, 1, 1, bias = False, padding_mode = 'reflect')
        self.d4_conv1 = nn.Conv2d(512, 512, 3, 1, 1, bias = False, padding_mode = 'reflect')
        
        self.d3_conv = nn.Conv2d(256, 256, 3, 1, 1, bias = False, padding_mode = 'reflect')
        self.d3_conv1 = nn.Conv2d(256, 256, 3, 1, 1, bias = False, padding_mode = 'reflect')
        
        
        self.d2_conv = nn.Conv2d(128, 128, 3, 1, 1, bias = False, padding_mode = 'reflect')
        self.d2_conv1 = nn.Conv2d(128, 128, 3, 1, 1, bias = False, padding_mode = 'reflect')
        
        
    def forward(self,x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        #print(d2.shape)
        d3 = self.down2(d2)
        #print(d3.shape)
        d4 = self.down3(d3)
        #print(d4.shape)
        d5 = self.down4(d4)
        #print(d5.shape)
        d6 = self.down5(d5)
        #print(d6.shape)
        d7 = self.down6(d6)
        #print(d7.shape)
        bottleneck = self.bottleneck(d7)
        # print(bottleneck.shape)
        up1 = self.up1(bottleneck)
        #print(up1.shape)
        
        ######## residual block1 ### 
        d7_ = F.relu(self.d7_conv(d7))
        d7_1 = self.d7_conv1(d7_)
        d7 = d7+d7_
        ############################
        
        
        up2 = self.up2(torch.cat([up1, d7], dim = 1))
        
        ######## residual block2 ###
        d6_ = F.relu(self.d6_conv(d6))
        d6_1 = self.d6_conv1(d6_)
        d6 = d6+d6_
        ###########################
        
        up3 = self.up3(torch.cat([up2, d6], dim = 1))
        
        ######## residual block3 ###
        d5_ = F.relu(self.d5_conv(d5))
        d5_1 = self.d5_conv1(d5_)
        d5 = d5+d5_
        ##############################
        
        
        up4 = self.up4(torch.cat([up3, d5], dim = 1))
        
        ######## residual block4 ###
        d4_ = F.relu(self.d4_conv(d4))
        d4_1 = self.d4_conv1(d4_)
        d4 = d4+d4_
        ############################
        
        up5 = self.up5(torch.cat([up4, d4], dim = 1))
        
        ######## residual block5 ###
        d3_ = F.relu(self.d3_conv(d3))
        d3_1 = self.d3_conv(d3_)
        d3 = d3+d3_
        ##############################
        
        up6 = self.up6(torch.cat([up5, d3], dim = 1))
        
        ######### residual block6 ###
        d2_ = F.relu(self.d2_conv(d2))
        d2_1 = self.d2_conv(d2_)
        d2 = d2+d2_
        ############################
        
        
        
        up7 = self.up7(torch.cat([up6, d2], dim = 1))
        
        
        return self.final_up(torch.cat([up7, d1], dim = 1)) 
z = torch.ones(1, 3, 256, 256)
Generator()(z).shape

class MapDataset(Dataset):
    def __init__(self,root_dir, list_patients1, list_patients2):
        super(MapDataset, self).__init__()
        self.root_dir = root_dir # train val folder 
        self.list_files_t1 = list_patients1 
        self.list_files_t2 = list_patients2 
    def __len__(self):
        return len(self.list_files_t1)*50 
    
    def __getitem__(self, idex):
        folder_idex = int(idex/50)
        image_idex = idex%50 
        
        folder_file1 = self.list_files_t1[folder_idex]
        image_file1 = os.listdir(self.root_dir+"/ixi-t1/image slice-T1/"+folder_file1)[image_idex]
        img_path1 = os.path.join(self.root_dir,'ixi-t1',"image slice-T1", folder_file1, image_file1)
        target_image = np.array(Image.open(img_path1))
        
        folder_file2 = self.list_files_t2[folder_idex]
        image_file2 = os.listdir(self.root_dir+"/ixit2-slices/image slice-T2/"+folder_file2)[image_idex]
        img_path2 = os.path.join(self.root_dir,'ixit2-slices',"image slice-T2", folder_file2, image_file2)
        input_image = np.array(Image.open(img_path2))
        
        #input_image = image[:, :600,:]
        #target_image = image[:, 600:, :]
        
        augmentations = both_transform(image=input_image, image0=target_image)
        input_image = augmentations["image"]
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        
        
        target_image = augmentations["image0"]
        target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
        target_image = cv2.rotate(target_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        #print(target_image.shape)

        input_image = transform_only_input(image=input_image)["image"]
        target_image = transform_only_mask(image=target_image)["image"]

        return input_image, target_image
      
def save_some_examples(gen, val_loader, epoch, folder):
    x, y = next(iter(val_loader))
    x, y = x.to(DEVICE), y.to(DEVICE)
    gen.eval()
    if not os.path.exists(folder):
        os.mkdir(folder)
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5  # remove normalization#
        save_image(y_fake, folder + f"/y_gen_{epoch}.png")
        save_image(x * 0.5 + 0.5, folder + f"/input_{epoch}.png")
        if epoch == 1:
            save_image(y * 0.5 + 0.5, folder + f"/label_{epoch}.png")
    gen.train()


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
        
device = "cuda" if torch.cuda.is_available() else "cpu"
disc = Discriminator().to(device)
gen = Generator().to(device)
opt_disc = optim.Adam(disc.parameters(), lr = 0.0002, betas = (0.5, 0.999))
opt_gen = optim.Adam(gen.parameters(), lr = 0.0002, betas = (0.5, 0.999))

BCE = nn.BCEWithLogitsLoss()
L1_LOSS = nn.L1Loss()

train_dataset = MapDataset(TRAIN_DIR, t1_sampled , t2_sampled)
train_loader = DataLoader(
    train_dataset, 
    batch_size = BATCH_SIZE,
    shuffle = True,
    num_workers = NUM_WORKERS
)


val_dataset = MapDataset(VAL_DIR, t1_sampled, t2_sampled)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

g_scaler = torch.cuda.amp.GradScaler()
d_scaler = torch.cuda.amp.GradScaler()

def train_fn(disc, gen, loader, opt_disc, opt_gen, l1_loss, bce, g_scaler, d_scaler):
    pbar = tqdm.tqdm(loader, leave = True)
    for idx,(x,y) in enumerate(pbar):
        #print(x.shape)
        #print(y.shape)
        x = x.to(device) # input image type 
        y = y.to(device) # target image type 

        # train discriminator 
        with torch.cuda.amp.autocast():
            y_fake = gen(x) # fake target generation
            D_real = disc(x,y) # disc pred with actual image 
            D_real_loss = bce(D_real, torch.ones_like(D_real))

            D_fake = disc(x, y_fake.detach())
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))

            D_loss = (D_fake_loss + D_real_loss)/2 

        opt_disc.zero_grad()
        # D_loss.backward()
        # opt_disc.step()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # train generator 
        with torch.cuda.amp.autocast():
            D_fake = disc(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1_loss(y_fake,y)*L1_LAMBDA
            G_loss = G_fake_loss + L1

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 10 == 0:
            pbar.set_postfix(
                D_real = torch.sigmoid(D_real).mean().item(),
                D_fake = torch.sigmoid(D_fake).mean().item(),
            )
        
SAVE_MODE = True 

for epoch in range(NUM_EPOCHS):
    train_fn(disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE, g_scaler, d_scaler)
    if SAVE_MODE and epoch %1 ==0: 
        save_checkpoint(gen, opt_gen, filename = CHECKPOINT_GEN)
        save_checkpoint(disc, opt_disc, filename = CHECKPOINT_DISC)
            
    save_some_examples(gen, val_loader, epoch, folder = 'evaluation') 
