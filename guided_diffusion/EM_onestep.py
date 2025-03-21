import numpy as np
import os
import torch
import torch.fft
# import cv2

def EM_Initial(IR):
    device = IR.device

    k1 = torch.tensor([[1,-1]]).to(device)
    k2 = torch.tensor([[1],[-1]]).to(device)

    fft_k1 = psf2otf(k1, np.shape(IR)).to(device)
    fft_k2 = psf2otf(k2, np.shape(IR)).to(device)
    fft_k1sq = torch.conj(fft_k1)*fft_k1.to(device)
    fft_k2sq = torch.conj(fft_k2)*fft_k2.to(device)

    C  = torch.ones_like(IR).to(device)
    D = torch.ones_like(IR).to(device)
    F2 = torch.zeros_like(IR).to(device)
    F1 = torch.zeros_like(IR).to(device)
    H  = torch.zeros_like(IR).to(device)
    tau_a = 1
    tau_b = 1
    HP={"C":C, "D":D, "F2":F2, "F1":F1, "H":H, "tau_a":tau_a, "tau_b":tau_b,
    "fft_k1":fft_k1, "fft_k2":fft_k2, "fft_k1sq":fft_k1sq, "fft_k2sq":fft_k2sq}
    return HP

def EM_InitialList(IR, K):
    device = IR.device
    k1 = torch.tensor([[1,-1]]).to(device)
    k2 = torch.tensor([[1],[-1]]).to(device)

    fft_k1 = psf2otf(k1, np.shape(IR)).to(device)
    fft_k2 = psf2otf(k2, np.shape(IR)).to(device)
    fft_k1sq = torch.conj(fft_k1)*fft_k1.to(device)
    fft_k2sq = torch.conj(fft_k2)*fft_k2.to(device)

    F2 = torch.zeros_like(IR).to(device)
    F1 = torch.zeros_like(IR).to(device)
    H  = torch.zeros_like(IR).to(device)
    tauList = 1*torch.ones(K).to(device)
    
    HP={"F2":F2, "F1":F1, "H":H, "tau_list":tauList,
    "fft_k1":fft_k1, "fft_k2":fft_k2, "fft_k1sq":fft_k1sq, "fft_k2sq":fft_k2sq}
    return HP

def EM_onestep(f_pre,I,V,HyperP,lamb=0.5,rho=0.01): 
    device = f_pre.device

    fft_k1 = HyperP["fft_k1"].to(device)
    fft_k2 = HyperP["fft_k2"].to(device)
    fft_k1sq = HyperP["fft_k1sq"].to(device)
    fft_k2sq = HyperP["fft_k2sq"].to(device)
    C  = HyperP["C"].to(device)
    D  = HyperP["D"].to(device)
    F2 = HyperP["F2"].to(device)
    F1 = HyperP["F1"].to(device)
    H  = HyperP["H"].to(device)
    tau_a = HyperP["tau_a"]#.to(device)
    tau_b = HyperP["tau_b"]#.to(device)

    LAMBDA =  lamb

    Y = I - V
    X = f_pre - V
    #e-step
    D = torch.sqrt(2/tau_b/(X**2+1e-6))
    C = torch.sqrt(2/tau_a/((Y-X+1e-6)**2))
    # print(f'cut nums {torch.sum(D>2*C)}, ratio is {torch.sum(D>2*C)/torch.numel(D)}')
    # print(f'D shape {D.shape}, C shape {C.shape}')
    D[D>2*C] = 2*C[D>2*C]
    RHO =rho # .5*(C+D)
    
    tau_b = 1./(D+1e-10)+tau_b/2
    tau_b = torch.mean(tau_b)
    tau_a = 1./(C+1e-10)+tau_a/2
    tau_a = torch.mean(tau_a)

    # m-step
    for s in range(1):
        H = prox_tv(Y-X, F1, F2, fft_k1, fft_k2, fft_k1sq, fft_k2sq)

        a1=torch.zeros_like(H)
        a1[:,:,:,:-1]=H[:,:,:,:-1]-H[:,:,:,1:]
        a1[:,:,:,-1]=H[:,:,:,-1]
        F1 = (RHO/(2*LAMBDA+RHO))*a1

        a2=torch.zeros_like(H)
        a2[:,:,:-1,:]=H[:,:,:-1,:]-H[:,:,1:,:]
        a2[:,:,-1,:]=H[:,:,-1,:]
        F2 = (RHO/(2*LAMBDA+RHO))*a2
        X = (2*C*Y+RHO*(Y-H))/(2*C+2*D+RHO)

    F = I-X

    return F,{"C":C, "D":D, "F2":F2, "F1":F1, "H":H, "tau_a":tau_a, "tau_b":tau_b,
    "fft_k1":fft_k1, "fft_k2":fft_k2, "fft_k1sq":fft_k1sq, "fft_k2sq":fft_k2sq}

def EM_onestepList(f_pre,IList,HyperP,lamb=0.5,rho=0.01):
    '''
    EM algorithm for one step
    f_pre: the initial image
    IList: the list of images, torch.tensor of shape (N, C, H, W),
    HyperP: the hyperparameters, a dict containing:
        - "fft_k1": the fft of the first kernel
        - "fft_k2": the fft of the second kernel
        - "fft_k1sq": the square of the fft of the first kernel
        - "fft_k2sq": the square of the fft of the second kernel
        - "C": the C matrix
        - "D": the D matrix
        - "F2": the F2 matrix
        - "F1": the F1 matrix
        - "H": the H matrix
        - "tau_list": list of tau values for each image
    lamb: the lambda value for the EM algorithm
    rho: the rho value for the EM algorithm
    return: the updated image and the updated hyperparameters
    '''
    device = f_pre.device

    fft_k1 = HyperP["fft_k1"].to(device)
    fft_k2 = HyperP["fft_k2"].to(device)
    fft_k1sq = HyperP["fft_k1sq"].to(device)
    fft_k2sq = HyperP["fft_k2sq"].to(device)
    
    # C  = HyperP["C"].to(device)
    # D  = HyperP["D"].to(device)
    
    F2 = HyperP["F2"].to(device)
    F1 = HyperP["F1"].to(device)
    H  = HyperP["H"].to(device)
    tau_list = HyperP["tau_list"].to(device)
    LAMBDA = lamb
    RHO =rho # .5*(C+D)
    
    #e-step
    Dlist = []
    for i in range(len(IList)):
        X = f_pre - IList[i]
        D = torch.sqrt(2/tau_list[i]/(X**2+1e-6))
        Dlist.append(D)        
        tau = 1./(D+1e-10)+tau_list[i]/2
        tau_list[i] = torch.mean(tau)
        
    Dlist = torch.cat(Dlist, dim = 0)
    
    # m-step
    for s in range(1):
        H = prox_tv(f_pre, F1, F2, fft_k1, fft_k2, fft_k1sq, fft_k2sq)

        a1=torch.zeros_like(H)
        a1[:,:,:,:-1]=H[:,:,:,:-1]-H[:,:,:,1:]
        a1[:,:,:,-1]=H[:,:,:,-1]
        F1 = (RHO/(2*LAMBDA+RHO))*a1

        a2=torch.zeros_like(H)
        a2[:,:,:-1,:]=H[:,:,:-1,:]-H[:,:,1:,:]
        a2[:,:,-1,:]=H[:,:,-1,:]
        F2 = (RHO/(2*LAMBDA+RHO))*a2
        F = (torch.sum(2 * Dlist * IList, dim = 0, keepdim = True) + RHO * H) / (torch.sum(2 * Dlist, dim = 0, keepdim = True) + RHO)

    return F,{"F2":F2, "F1":F1, "H":H, "tau_list": tau_list,
    "fft_k1":fft_k1, "fft_k2":fft_k2, "fft_k1sq":fft_k1sq, "fft_k2sq":fft_k2sq, "Dlist":Dlist}

def prox_tv(X, F1, F2, fft_k1, fft_k2, fft_k1sq, fft_k2sq):
    fft_X = torch.fft.fft2(X)
    fft_F1 = torch.fft.fft2(F1)
    fft_F2 = torch.fft.fft2(F2)

    H = fft_X + torch.conj(fft_k1)*fft_F1 + torch.conj(fft_k2)*fft_F2
    H = H/( 1+fft_k1sq+fft_k2sq)
    H = torch.real(torch.fft.ifft2(H))
    return H

def psf2otf(psf, outSize):
    psfSize = torch.tensor(psf.shape)
    outSize = torch.tensor(outSize[-2:])
    padSize = outSize - psfSize
    psf=torch.nn.functional.pad(psf,(0, padSize[1],0, padSize[0]), 'constant')

    for i in range(len(psfSize)):
        psf = torch.roll(psf, -int(psfSize[i] / 2), i)
    otf = torch.fft.fftn(psf)
    nElem = torch.prod(psfSize)
    nOps = 0
    for k in range(len(psfSize)):
        nffts = nElem / psfSize[k]
        nOps = nOps + psfSize[k] * torch.log2(psfSize[k]) * nffts
    if torch.max(torch.abs(torch.imag(otf))) / torch.max(torch.abs(otf)) <= nOps * torch.finfo(torch.float32).eps:
        otf = torch.real(otf)
    return otf
    
