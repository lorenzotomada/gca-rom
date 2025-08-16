#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/fpichi/gca-rom/blob/main/notebook/10_stokes.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[1]:


# Install PyTorch
import torch


# In[2]:


# Install PyG
import torch_geometric


# In[3]:


# Clone and import gca-rom
import sys
sys.path.append('./..')

from gca_rom import network, pde, loader, plotting, preprocessing, training, initialization, testing, error, gui


# In[4]:


import numpy as np
from itertools import product
from IPython.display import HTML


# # Define PDE problem

# In[5]:


problem_name, variable, mu_space, n_param, dim_pde, n_comp = pde.problem(13)
argv = gui.hyperparameters_selection(problem_name, variable, n_param, n_comp)
argv[5] = 75
argv[8] = 15
HyperParams = network.HyperParams(argv)
HyperParams.__dict__


# # Initialize device and set reproducibility

# In[6]:


device = initialization.set_device()
initialization.set_reproducibility(HyperParams)
initialization.set_path(HyperParams)


# # Load dataset

# In[7]:


dataset_dir = '../dataset/'+problem_name+'_unstructured.mat'
dataset = loader.LoadDataset(dataset_dir, variable, dim_pde, n_comp)

time = mu_space[-1]
n_time = len(time)
n_snap = np.prod([len(mu_space[i]) for i in range(n_param)])
n_sim = int(n_snap/n_time)

n_snap2keep = n_time
print(f'N snaps to keep: {n_time}')
#dataset, mu_space = preprocessing.shrink_dataset(dataset, mu_space, n_sim, n_snap2keep, n_comp)

params = torch.tensor(np.array(list(product(*mu_space))))
params = params.to(device)

dataset, params, mu_space = preprocessing.delete_initial_condition(dataset, params, mu_space, n_comp, n_snap2keep)

graph_loader, train_loader, test_loader, \
    val_loader, scaler_all, scaler_test, xyz, VAR_all, VAR_test, \
        train_trajectories, test_trajectories = preprocessing.graphs_dataset(dataset, HyperParams, n_sim)


# # Define the architecture

# In[8]:


model = network.Net(HyperParams)
model = model.to(device)
torch.set_default_dtype(torch.float32)
optimizer = torch.optim.Adam(model.parameters(), lr=HyperParams.learning_rate, weight_decay=HyperParams.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=HyperParams.miles, gamma=HyperParams.gamma)


# # Train or load a pre-trained network

# In[9]:

VAR_all = VAR_all.to('cpu')
VAR_test = VAR_test.to('cpu')

try:
    model.load_state_dict(torch.load(HyperParams.net_dir+HyperParams.net_name+HyperParams.net_run+'.pt'))
    print('Loading saved network')
except FileNotFoundError:
    print('Training network')
    training.train(model, optimizer, device, scheduler, params, train_loader, test_loader, train_trajectories, test_trajectories, HyperParams)


# # Evaluate the model

# In[10]:

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Trainable parameters in net:", count_trainable_params(model))
print("Trainable params in maptovec: ", count_trainable_params(model.maptovec))
# # Train or load a pre-trained network


model.to("cpu")
params = params.to("cpu")
vars = "GCA-ROM"
results, latents_map, latents_gca = testing.evaluate(VAR_all, model, graph_loader, params, HyperParams, range(params.shape[0]))


# # Plot the results

# In[11]:


SAMPLE = 10
plotting.plot_sample(HyperParams, mu_space, params, train_trajectories, test_trajectories)
plotting.plot_loss(HyperParams)
plotting.plot_latent(HyperParams, latents_map, latents_gca)
plotting.plot_latent_time(HyperParams, SAMPLE, latents_map, mu_space, params, n_sim)

plotting.plot_error(results, VAR_all, scaler_all, HyperParams, mu_space, params, train_trajectories, vars)
plotting.plot_error_2d(results, VAR_all, scaler_all, HyperParams, mu_space, params, train_trajectories, vars, p1=0, p2=1)


# In[12]:


N = 3
snapshots = np.arange(params.shape[0]).tolist()
np.random.shuffle(snapshots)
for SNAP in snapshots[0:N]:
    plotting.plot_fields(SNAP, results, scaler_all, HyperParams, dataset, xyz, params)
    plotting.plot_error_fields(SNAP, results, VAR_all, scaler_all, HyperParams, dataset, xyz, params)


# In[13]:


#anim = plotting.create_animation(SAMPLE, VAR_all, scaler_all, HyperParams, dataset, xyz, params, n_sim)
#HTML(anim.to_jshtml())


# # Print the errors on the testing set

# In[14]:


results_test, _, _ = testing.evaluate(VAR_test, model, val_loader, params, HyperParams, test_trajectories)

error_abs, norm = error.compute_error(results_test, VAR_test, scaler_test, HyperParams)
error.print_error(error_abs, norm, vars)
error.save_error(error_abs, norm, HyperParams, vars)

np.save("errors_MH.npy", error_abs)
np.save("norms_MH.npy", norm)


plotting.plot_comparison_fields(results, VAR_all, scaler_all, HyperParams, dataset, xyz, params)
plotting.plot_error_3d(results_test, VAR_test, scaler_test, HyperParams, mu_space, params, train_trajectories, vars, test_trajectories=test_trajectories)

from gca_rom import scaling

Z = scaling.inverse_scaling(results, scaler_all, HyperParams.scaling_type)
Z = Z.numpy()

np.save(f"scaled_output_gca_{HyperParams.net_name}_bottleneck_{HyperParams.bottleneck_dim}.npy", Z)
