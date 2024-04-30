import argparse
import torch
import utils
import datetime
import os
import pickle
import wandb
import numpy as np
import logging

from torch.utils import data
import torch.nn.functional as F

# from torch.utils.tensorboard import SummaryWriter
import modules


parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=1024,
                    help='Batch size.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of training epochs.')
parser.add_argument('--learning-rate', type=float, default=5e-4,
                    help='Learning rate.')

parser.add_argument('--sigma', type=float, default=0.5,
                    help='Energy scale.')
parser.add_argument('--hinge', type=float, default=1.,
                    help='Hinge threshold parameter.')

parser.add_argument('--hidden-dim', type=int, default=512,
                    help='Number of hidden units in transition MLP.')
parser.add_argument('--embedding-dim', type=int, default=16,
                    help='Dimensionality of embedding.')
parser.add_argument('--action-dim', type=int, default=4,
                    help='Dimensionality of action space.')
parser.add_argument('--num-objects', type=int, default=1,
                    help='Number of object slots in model.')
parser.add_argument('--ignore-action', action='store_true', default=False,
                    help='Ignore action in GNN transition model.')
parser.add_argument('--copy-action', action='store_true', default=False,
                    help='Apply same action to all object slots.')

parser.add_argument('--decoder', action='store_true', default=False,
                    help='Train model using decoder and pixel-based loss.')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disable CUDA training.')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed (default: 42).')
parser.add_argument('--log-interval', type=int, default=20,
                    help='How many batches to wait before logging'
                         'training status.')
parser.add_argument('--dataset', type=str,
                    default='data/shapes_train.h5',
                    help='Path to replay buffer.')
parser.add_argument('--name', type=str, default='none',
                    help='Experiment name.')
parser.add_argument('--save-folder', type=str,
                    default='checkpoints',
                    help='Path to checkpoints.')


# start a new wandb run to track this script



args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# wandb.init(
#     # set the wandb project where this run will be logged
#     project="pgm",
#     config = vars(args)
# )
print("Started Script with CUDA: ", args.cuda)

now = datetime.datetime.now()
timestamp = now.isoformat()

if args.name == 'none':
    exp_name = timestamp
else:
    exp_name = args.name

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

exp_counter = 0
save_folder = '{}/{}/'.format(args.save_folder, exp_name)

if not os.path.exists(save_folder):
    os.makedirs(save_folder)
meta_file = os.path.join(save_folder, 'metadata.pkl')
model_file = os.path.join(save_folder, 'model.pt')
log_file = os.path.join(save_folder, 'log.txt')

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
logger.addHandler(logging.FileHandler(log_file, 'a'))
print = logger.info

pickle.dump({'args': args}, open(meta_file, "wb"))

device = torch.device('cuda' if args.cuda else 'cpu')

print("Loading data...")
dataset = utils.StateTransitionsDataset(
    hdf5_file=args.dataset)
train_loader = data.DataLoader(
    dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
print('Data loaded!')

# Get data sample
obs = train_loader.__iter__().next()[0]
input_shape = obs[0].size()

print("Creating model...")
model = modules.ContrastiveSWMwithFrozenVAE(
    embedding_dim=args.embedding_dim,
    hidden_dim=args.hidden_dim,
    action_dim=args.action_dim,
    input_dims=input_shape,
    num_objects=args.num_objects,
    sigma=args.sigma,
    hinge=args.hinge,
    ignore_action=args.ignore_action,
    copy_action=args.copy_action,
    ).to(device)
print("Model created")

# optimizer only for transition model, rest is frozen
optimizer = torch.optim.Adam(
    model.transition_model.parameters(),
    lr=args.learning_rate)

# Train model.
print('Starting model training...')
step = 0
best_loss = 1e9
# writer = SummaryWriter()

for epoch in range(1, args.epochs + 1):
    model.train()
    train_loss = 0
    avg_mse_latent = 0
    for batch_idx, data_batch in enumerate(train_loader):
        data_batch = [tensor.to(device) for tensor in data_batch]
        optimizer.zero_grad()
        obs, action, next_obs = data_batch
        state = model.vae.get_encoding(obs).unsqueeze(1)
        rec = model.vae.decode(state.squeeze(1))
        next_state_pred = state + model.transition_model(state, action)
        next_rec = model.vae.decode(next_state_pred.squeeze(1))
        if args.decoder:
            loss = F.mse_loss(
                rec, obs, reduction='sum') / obs.size(0)
            next_loss = F.mse_loss(
                next_rec, next_obs,
                reduction='sum') / obs.size(0)
            loss += next_loss
        else:
            loss = model.contrastive_loss(*data_batch)

        avg_mse_latent += F.mse_loss(
            state.squeeze(1), next_state_pred.squeeze(1),
            reduction='sum').item() / obs.size(0)
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print(
                'Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data_batch[0]),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item() / len(data_batch[0])))

        step += 1

    avg_loss = train_loss / len(train_loader.dataset)
    avg_mse_latent /= len(train_loader.dataset)
    # writer.add_scalar("Loss/train", avg_loss, epoch)
    # wandb.log({"avg_loss": loss})
    print('====> Epoch: {} MSE_Recon: {:.6f} MSE_Latent: {:.6f}'.format(
        epoch, avg_loss, avg_mse_latent))

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), model_file)
