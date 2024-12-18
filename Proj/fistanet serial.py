import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import VGG16_Weights

torch.autograd.set_detect_anomaly(True)
import os
from time import time
from torch.nn import init
from torch.utils.data import Dataset, random_split
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
from skimage.metrics import structural_similarity as ssim3d
from torch.utils.data import DataLoader

from tqdm import tqdm

from config import START_EPOCH, END_EPOCH, LEARNING_RATE, LAYER_NUM, NUM_FILE, BATCH_SIZE,\
                    MODEL_DIR, DATA_DIR, LOG_DIR, PSF_DIR, LAMBDA_STEP, SOFT_THRESHOLD, LAMBDA_PERC

start_epoch = START_EPOCH
end_epoch = END_EPOCH
learning_rate = LEARNING_RATE
layer_num = LAYER_NUM
lambda_perc = 0.001
lambda_reg = 1e-10

device = torch.device("cuda:0")

model_dir = "%s/OPFISTA_Net_reg10pl3_layer_%d_lr_%.4f" % (MODEL_DIR, layer_num, learning_rate)

log_file_name = "%s/NDLog_CS_ISTA_Net_layer_%d_lr_%.4f.txt" % (LOG_DIR, layer_num, learning_rate)

def mapping_mse(y_pred, y_true):
    diff = y_pred - y_true
    return torch.mean(diff ** 2)

class CustomDataset(Dataset):
    def __init__(self, data_dir, num_files):
        self.data_dir = data_dir
        self.num_files = num_files

    def __len__(self):
        return self.num_files
    
    def __getitem__(self, idx):

        sd = f'{self.data_dir}/mimic_b80p20_scanning_data{idx + 1}.nii'
        gt = f'{self.data_dir}/mimic_b80p20_ground_truth{idx + 1}.nii'
        scanning_data = nib.load(sd).get_fdata()
        ground_truth = nib.load(gt).get_fdata()

        return scanning_data, ground_truth

num_files = NUM_FILE
dataset = CustomDataset(data_dir=DATA_DIR, num_files=num_files)
data_num = len(dataset)
train_size = 0.8*data_num
valid_size = 0.2*data_num
train_set, valid_set = random_split(dataset, [int(train_size), int(valid_size)])

batch_size = BATCH_SIZE
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True)


if not os.path.exists(model_dir):
    os.makedirs(model_dir)

import torch
import torch.fft as fft

def convolution_with_psf_3d(A, psf):
    # Get shapes of input tensors
    A_shape = torch.tensor(A.shape)
    psf_shape = torch.tensor(psf.shape)

    # Create zero-filled tensors
    psf1 = torch.zeros_like(A, dtype=psf.dtype, device=A.device)
    B = torch.zeros_like(A, dtype=A.dtype, device=A.device)

    psf1[:psf_shape[0], :psf_shape[1], :psf_shape[2]] = psf

    center_A = [(size) // 2 for size in (A_shape - psf_shape)]

    psf1 = torch.roll(psf1, center_A, dims=(0, 1, 2))

    # Fast convolution
    A1_fft = fft.fftn(A)
    psf1_fft = fft.fftn(psf1)
    B = fft.ifftn(A1_fft * psf1_fft)
    B = fft.fftshift(B)

    return B


## model
def H(input_tensor, psf):
    
    input_size = input_tensor.size()
    psf_size = psf.size()
 
    assert len(input_size) == 4 and len(psf_size) == 5, "Input and PSF must be 3D tensors"


    new_tensor = input_tensor.unsqueeze(1)
    # repeat 3 times
    new_tensor_repeated = new_tensor.repeat(1, psf_size[1], 1, 1, 1) 
    new_tensor_size = new_tensor_repeated.size()
    new_tensor_repeated = new_tensor_repeated.to(device)

    # conv
    conv_result = torch.zeros_like(new_tensor_repeated, dtype=input_tensor.dtype, device=input_tensor.device)
    for i in range(new_tensor_size[0]):
        for j in range(new_tensor_size[1]):
            A = new_tensor_repeated[i,j,:,:,:]
            B = psf[i,j,:,:,:]
            conv_result[i,j,:,:,:] = convolution_with_psf_3d(A, B)

    return conv_result


def HT(input_tensor, psf):

    input_size = input_tensor.size()
    psf_size = psf.size()

    assert len(input_size) == 5 and len(psf_size) == 5, "Input and PSF must be 3D tensors"
    assert input_size[:2] == psf_size[:2], "First two dimensions of input and PSF must match"

    # 初始化存储结果的 tensor
    result = torch.zeros(input_size[0], input_size[2], input_size[3], input_size[4])

    # 第一个维度翻转
    psf_flipped = psf.flip(2)
    # 第二个维度翻转
    psf_flipped = psf_flipped.flip(3)
    # 第三个维度翻转
    psf_flipped = psf_flipped.flip(4)

    # 使用 conv2d 进行相关操作
    conv_result = torch.zeros(input_size, dtype=input_tensor.dtype, device=input_tensor.device)
    for i in range(input_size[0]):
        for j in range(input_size[1]):
            A = input_tensor[i,j,:,:,:]
            B = psf_flipped[i,j,:,:,:]
            conv_result[i,j,:,:,:] = convolution_with_psf_3d(A, B)

    # 将相关结果叠加在一起
    result = torch.sum(conv_result, dim=1)
    
    return result

class BasicBlock(torch.nn.Module):
    def __init__(self):
        super(BasicBlock, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor(LAMBDA_STEP))
        self.soft_thr = nn.Parameter(torch.Tensor(SOFT_THRESHOLD))

        self.conv_D = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 1, 3, 3, 3)))

        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3, 3)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3, 3)))
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3, 3)))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3, 3)))

        self.conv_G = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 32, 3, 3 ,3)))

    def forward(self, x, psf, b):

        if len(x.size()) == 3:
        # 在第一维增加一个新的维度
            x = x.unsqueeze(0)
        x = x - self.lambda_step * HT((H(x,psf)-b),psf)
        # print(self.lambda_step)

        x_input = x.unsqueeze(1).float()

        x_D = F.conv3d(x_input, self.conv_D, padding=1)

        x = F.conv3d(x_D, self.conv1_forward, padding=1)
        x = F.relu(x)
        x_forward = F.conv3d(x, self.conv2_forward, padding=1)

        x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr))

        x = F.conv3d(x, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_backward = F.conv3d(x, self.conv2_backward, padding=1)

        x_G = F.conv3d(x_backward, self.conv_G, padding=1)
  
        # prediction output (skip connection); non-negative output
        x_pred = F.relu(x_input + x_G)

        symloss = x_pred - x_input
        x_pred = x_pred.squeeze()

        return [x_pred, symloss]
    
psf_file = PSF_DIR
psf = nib.load(psf_file)

# 获取图像数据
psf = psf.get_fdata()
selected_psf = psf[[0, 3, 6], ...]
psf = torch.from_numpy(selected_psf)
psf = psf.unsqueeze(0)  
psf = psf.repeat(batch_size,1, 1, 1, 1) 

# Define FISTA-Net
class FISTANet(torch.nn.Module):
    def __init__(self, LayerNo):
        super(FISTANet, self).__init__()
        onelayer = []
        onelayer = []
        self.LayerNo = LayerNo

        for i in range(LayerNo):
            onelayer.append(BasicBlock())

        self.fcs = nn.ModuleList(onelayer)
        
        # two-step update weight
        self.w_rho = nn.Parameter(torch.Tensor([0.5]))
        self.b_rho = nn.Parameter(torch.Tensor([0]))
        self.psf = nn.Parameter(psf)

        self.Sp = nn.Softplus()

    def forward(self, p_measured):

        p_measured_size = p_measured.size()

        x0 = torch.ones(p_measured_size[0], p_measured_size[2], p_measured_size[3], p_measured_size[4]).to(p_measured.device)
    
        xold = x0
        y = xold 
        layers_sym = []   # for computing symmetric loss
        
        for i in range(self.LayerNo):
            [xnew, layer_sym] = self.fcs[i](y, self.psf, p_measured)

            rho_ = (self.Sp(self.w_rho * i + self.b_rho) -  self.Sp(self.b_rho)) / self.Sp(self.w_rho * i + self.b_rho)
            y = xnew + rho_ * (xnew - xold) # two-step update
            xold = xnew

            layers_sym.append(layer_sym)
        
        x_final = xold
        
        return [x_final, layers_sym]
    

class PerceptualLoss(nn.Module):
    def __init__(self, feature_layers):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(weights=VGG16_Weights.DEFAULT).features
        self.features = nn.ModuleList([vgg[i] for i in feature_layers]).eval()
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, output, target):
        output_features = []
        target_features = []
        for feature in self.features:
            output = feature(output)
            target = feature(target)
            output_features.append(output)
            target_features.append(target)

        loss = 0
        for o, t in zip(output_features, target_features):
            loss += torch.nn.functional.l1_loss(o, t)
        return loss * lambda_perc
    

model = FISTANet(layer_num)
model = model.to(device)

perceptual_loss = PerceptualLoss(feature_layers=[3, 8, 15])  # VGG的卷积层索引
perceptual_loss = perceptual_loss.to(device)

model.train()

optimizer = torch.optim.Adam([
            {'params': model.fcs.parameters()}, 
            {'params': model.w_rho, 'lr': 0.0001},
            {'params': model.b_rho, 'lr': 0.0001},
            {'params': model.psf, 'lr': 1e-7},], 
            lr=learning_rate)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

for epoch_i in range(start_epoch, end_epoch):
    model.train()  # Set the model to training mode

    # Training phase
    for p_measured, labels in tqdm(train_loader):
        p_measured = p_measured.to(device)
        labels = labels.to(device)

        x_output, loss_layers_sym = model(p_measured)
        
        # Compute loss
        loss_discrepancy = torch.mean(mapping_mse(x_output, labels))
        loss_constraint = torch.mean(torch.pow(loss_layers_sym[0], 2))
        for k in range(layer_num - 1):
            loss_constraint += torch.mean(torch.pow(loss_layers_sym[k+1], 2))

        gamma = torch.Tensor([0.01]).to(device)

        x_output_numpy = x_output.squeeze().cpu().detach().numpy()
        labels_numpy = labels.squeeze().cpu().detach().numpy()   
        total_ssim = 0.0
        for i in range(batch_size):
            # 计算 SSIM
            ssim_index = ssim3d(x_output_numpy[i], labels_numpy[i], data_range=1.0)
            total_ssim += ssim_index
        
        ssim_loss = 1- total_ssim / batch_size

        loss_perceptual = perceptual_loss(x_output, labels)

        loss_all = loss_discrepancy + torch.mul(gamma, loss_constraint) + loss_perceptual + lambda_reg * torch.norm(x_output, p=2)

        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()

        print(f"[{epoch_i}/{end_epoch}] Training Loss: {loss_all.item():.5f}, MSE Loss(*10): {loss_discrepancy*10:.5f}, Constraint Loss:{loss_constraint:.5f} SSIM Loss: {ssim_loss:.5f}")


    # Validation phase
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for p_measured, labels in valid_loader:
            p_measured = p_measured.to(device)
            labels = labels.to(device)

            x_output, _ = model(p_measured)
            loss = torch.mean(mapping_mse(x_output, labels))

            print(f"[{epoch_i}/{end_epoch}] Validation Loss: {loss.item():.5f}")

    scheduler.step()
    # Save model periodically
    if epoch_i % 50 == 0:
        torch.save(model.state_dict(), f"{model_dir}/net_params_layer_10_p_r{epoch_i}.pkl")