#%%
import torch
import pdb
import matplotlib.pyplot as plt

# Load data
effective_CLS = torch.load("/mnt/c/Users/kaise/Desktop/researchData/effective_CLS_MRPC.pt")
standard_CLS = torch.load("/mnt/c/Users/kaise/Desktop/researchData/standard_CLS_MRPC.pt")

effective_SEP = torch.load("/mnt/c/Users/kaise/Desktop/researchData/effectiveSEP_MRPC.pt")
standard_SEP = torch.load("/mnt/c/Users/kaise/Desktop/researchData/standardSEP_MRPC.pt")

diff_CLS = torch.abs(effective_CLS - standard_CLS)
diff_SEP = torch.abs(effective_SEP - standard_SEP)

# Picturing
plt.figure()
plt.imshow(diff_CLS.numpy(), vmin=0, vmax=1.0, cmap='Greens')
plt.colorbar()
plt.ylabel("Layer")
plt.xlabel("Head")
plt.title("[CLS] difference(absolute value)")
plt.savefig('/mnt/c/Users/kaise/Desktop/researchData/CLS_Difference_MRPC.png')

plt.figure()
plt.imshow(diff_SEP.numpy(), vmin=0, vmax=1.0, cmap='Greens')
plt.colorbar()
plt.ylabel("Layer")
plt.xlabel("Head")
plt.title("[SEP] difference(absolute value)")
plt.savefig('/mnt/c/Users/kaise/Desktop/researchData/SEP_Difference_MRPC.png')

# Compute norm
print(torch.norm(diff_CLS))
print(torch.norm(diff_SEP))



