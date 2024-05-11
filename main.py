import torch
import numpy as np
import optuna
from fvcore.nn import FlopCountAnalysis
import torchvision.transforms as transforms
from loadfiles import load_datas
import torch.optim as optim
from torch import nn
from torchviz import make_dot
import matplotlib.pyplot as plt
import datetime
from torch.utils.data import Dataset, DataLoader
from loadfiles import LengthEstimator
from sklearn.model_selection import train_test_split

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.set_default_dtype(torch.float64)
# Step 1: Generate Dataset
def get_input(directory):
    """returned data type list: # of files x
                            nparray: 500 x [x1, y1, x2, y2, x3, y3, true_lenghts]"""
    all_data = load_datas(directory)
    print(len(all_data))
    return all_data[:-2], all_data[-2:]

def create_bestmodel(in_size, out_size, params):
    """To create a model with parameters returned by optuna study. """
    n_layers = params['num_layers']
    layers = []

    for i in range(n_layers):
        n_units = params[f"n_units_l{i}"]
        layer = nn.Linear(in_size, n_units)
        nn.init.kaiming_normal_(layer.weight)
        layers.append(layer)
        layers.append(nn.ReLU())
        in_size = n_units
    layers.append(nn.Linear(in_size, out_size))

    return nn.Sequential(*layers)

def create_model(trial, in_size, out_size, n_layers):

    layers = []
    for i in range(n_layers):
        n_units = trial.suggest_int("n_units_l{}".format(i), 32, 240, log=True)
        layer = nn.Linear(in_size, n_units)
        nn.init.kaiming_normal_(layer.weight)
        layers.append(layer)
        layers.append(nn.ReLU())
        in_size = n_units
    layers.append(nn.Linear(in_size, out_size))

    return nn.Sequential(*layers)

def mape(targets, outs):
    if torch.is_tensor(targets):
        if torch.norm(targets) >= torch.norm(outs):
            mask = targets == 0
            targets[mask] = 0.01
            abs_perc_error = torch.abs((targets - outs) / targets) * 100
            mape = torch.mean(abs_perc_error)
        else:
            mask = outs == 0
            outs[mask] = 0.01
            abs_perc_error = torch.abs((targets - outs) / outs) * 100
            mape = torch.mean(abs_perc_error)
        return mape
    if isinstance(outs, float):
        mape = abs((targets-outs)/max(targets, outs))*100
        return mape

def plotloss(loss, name):
    plt.plot(loss, label=name)
    plt.xlabel('Batch')
    plt.ylabel('Percentage Loss')
    plt.title(f'{name}')
    plt.legend()
    plt.grid(True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(f"Evaluation Loss {timestamp}")
    plt.close()

def objective(trial):
    # Categorical parameter
    #optimizer = trial.suggest_categorical("optimizer", ["MomentumSGD", "Adam"])
    all_data = LengthEstimator("testsamples1")
    train_data, test_data = train_test_split(all_data, test_size=0.2, random_state=42)
    batch_size = trial.suggest_int("batch_size", 25, 50)
    train_dl = DataLoader(train_data, batch_size, shuffle=True)
    test_dl = DataLoader(test_data, batch_size, shuffle=False)
    in_size = 6
    out_size = 1
    print(in_size)
    num_layers = trial.suggest_int("num_layers", 5, 30)
    # Integer parameter (log)
    #num_channels = trial.suggest_int("num_channels", 45, 750, log=True)
    # Floating point parameter (log)
    learning_rate = trial.suggest_float("learning_rate", 1e-8, 1e-5, log=True)
    model = create_model(trial, in_size, out_size, num_layers).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    train_model(model, optimizer, batch_size, train_dl)
    avg_loss = eval_model(model, test_dl)
    return avg_loss

def normalize(data):
    """inputs are [:,:15]"""
    mean = data[:, :6].mean(dim=1, keepdim=True)
    std = data[:, :6].std(dim=1, keepdim=True)
    norm_data = (data[:, :6] - mean) / std
    mask = np.isnan(norm_data).bool()
    norm_data[mask] = 0
    return norm_data.to(torch.float64), mean, std

def denormalize(data, mean, std):
    denorm_data = data * std + mean
    return denorm_data.to(torch.float64)

def train_model(model, optimizer, batch_size, train_dl):
    model.train(True)
    sample_count = 0
    losses = []
    for epoch in range(100):
        run_loss = 0.0
        for batch in train_dl:
            batch[:, :6], mean, std = normalize(batch)
            input_tensor = batch[:, :6]
            target = batch[:, -1]
            target.reshape(-1,1)
            output = model(input_tensor)
            output = denormalize(output, mean, std)
            loss = torch.linalg.norm(target-output)/len(batch)
            losses.append(loss.item())
            run_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sample_count += batch_size
            if sample_count % 300 == 0:
                print(f'\t\t\t Target \t Output  \n')
                for target1, pose_out in zip(target.data.numpy(), output.data.numpy()):
                    print(f'{target1} | {pose_out} |')
                print("\t Loss:", loss.item())

    plotloss(losses, "Training Loss")

def eval_model(model, test_dl):
    model.eval()
    loss = 0
    cnt = 0
    losses = []
    with torch.no_grad():
        for batch in test_dl:
            batch[:, :6], mean, std = normalize(batch)
            input_tensor = batch[:, :6]
            target = batch[:, -1]
            target.reshape(-1, 1)
            output = model(input_tensor)
            output = denormalize(output, mean, std)
            batch_size = len(batch)
            perc_loss = mape(target, output)/batch_size
            cnt += len(batch)
            losses.append(perc_loss)
            loss += perc_loss
            print(f'Predicted length for New Points: {output.data.numpy()} and true length: {target.data.numpy()}')

    avg_err = loss / cnt
    plotloss(losses, "Evaluation Loss")
    return avg_err


"""
# Step 2: Neural Network Model
class PathLengthEstimator(nn.Module):
    def __init__(self, trial, in_size, out_size):
        super().__init__()
        self.seq_layers = create_model(trial, in_size, out_size)

    def forward(self, x):
        #x = F.relu(self.nl1(self.l1(x)))
        #x = self.l2(self.nl2(x))
        #x = F.relu(self.l4(self.l3(x)))
        return self.seq_layers(x)
"""

if __name__ == '__main__':
    # Generate Dataset
    """
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)

    print("Number of finished trials: ", len(study.trials))
    print(f"Number of trials on the Pareto front: {len(study.best_trials)}")

    trial_with_highest_accuracy = min(study.best_trials, key=lambda t: t.values[0])
    print(f"Trial with highest accuracy: ")
    print(f"\tnumber: {trial_with_highest_accuracy.number}")
    print(f"\tparams: {trial_with_highest_accuracy.params}")
    print(f"\tvalues: {trial_with_highest_accuracy.values}")
    optuna.visualization.plot_param_importances(
        study, target=lambda t: t.values[0], target_name="avg_loss"
    )
    """

    # Initialize Best Model
    params = {'batch_size': 49, 'num_layers': 9, 'learning_rate': 6.929712835628966e-07, 'n_units_l0': 183,
             'n_units_l1': 179, 'n_units_l2': 66, 'n_units_l3': 164, 'n_units_l4': 64, 'n_units_l5': 145,
             'n_units_l6': 63, 'n_units_l7': 63, 'n_units_l8': 81}

    learning_rate = params["learning_rate"]
    batch_size = params["batch_size"]
    all_data = LengthEstimator("testsamples1")
    train_data, test_data = train_test_split(all_data, test_size=0.2, random_state=42)
    in_size = 6
    out_size = 1
    model = create_bestmodel(in_size, out_size, params).to(DEVICE)
    #model.load_state_dict(torch.load("length_over3points.pth"))
    train_dl = DataLoader(train_data, batch_size, shuffle=True)
    test_dl = DataLoader(test_data, batch_size, shuffle=False)
    # Train Model
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_model(model, optimizer, batch_size, train_dl)
    perc_err = eval_model(model, test_dl)
    print("Evaluation Perc. Error: ", perc_err)
    #save the model:
    torch.save(model.state_dict(), "length_over3points.pth")
    print("saved.")
    #exit(0)"""

