import random
import datetime
import torch
import numpy as np
import optuna
from optuna.visualization import plot_optimization_history
from fvcore.nn import FlopCountAnalysis
import torchvision.transforms as transforms
from loadfiles import TrajectoryOptimizationDataset
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
from torchviz import make_dot
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.set_default_dtype(torch.float64)
#writer = SummaryWriter('runs/Traj_Optim_Trials')


class mymodel(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.l1 = nn.Linear(in_size, int(in_size/2))
        nn.init.kaiming_normal_(self.l1.weight)
        self.r1 = nn.ReLU()
        self.l2 = nn.Linear(int(in_size/2), 4)
        nn.init.kaiming_normal_(self.l2.weight)

    def forward(self, x):
        outs = self.l1(x)
        return outs[:, :3], outs[:, 3]


def create_bestmodel(in_size, out_size, params):
    """To create a model with parameters returned by optuna study. """
    n_layers = params['num_layers']
    layers = []

    for i in range(n_layers):
        n_units = params[f"n_units_l{i}"]
        layer = nn.Linear(in_size, n_units)
        nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
        layers.append(layer)
        layers.append(nn.ReLU())
        in_size = n_units
    layers.append(mymodel(in_size, out_size))

    return nn.Sequential(*layers)


def create_model(trial, in_size, out_size, n_layers):
    """To create a model for optimization trials"""
    layers = []
    for i in range(n_layers):
        n_units = trial.suggest_int("n_units_l{}".format(i), 32, 256, log=True)
        layer = nn.Linear(in_size, n_units)
        nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
        layers.append(layer)
        layers.append(nn.ReLU())
        in_size = n_units
    layers.append(mymodel(in_size, out_size))

    return nn.Sequential(*layers)


def objective(trial):
    # Categorical parameter
    # optimizer = trial.suggest_categorical("optimizer", ["MomentumSGD", "Adam"])
    all_data = TrajectoryOptimizationDataset("samples2")
    train_data, test_data = train_test_split(all_data, test_size=0.2, random_state=42)
    # train_data = torch.utils.data.Subset(all_data, train_inds)
    # test_data = torch.utils.data.Subset(all_data, test_inds)
    batch_size = trial.suggest_int("batch_size", 25, 50)
    train_dl = DataLoader(train_data, batch_size, shuffle=True)
    test_dl = DataLoader(test_data, batch_size, shuffle=False)
    in_size = len(train_data[0]) - 4
    out_size = 4
    num_layers = trial.suggest_int("num_layers", 25, 300)
    n_epochs = trial.suggest_int("num_epochs", 3, 5)
    # Integer parameter (log)
    # num_channels = trial.suggest_int("num_channels", 45, 750, log=True)
    # Floating point parameter (log)
    learning_rate = trial.suggest_float("learning_rate", 6.357916695682364e-09, 1.4649050720229706e-04, log=True)
    model = create_model(trial, in_size, out_size, num_layers).to(DEVICE)
    """loss_fnc = trial.suggest_categorical("loss_function", ["l1", "huber", "mse", "smooth_loss"])
    if loss_fnc == "l1":
        loss_fnc = nn.L1Loss(reduction='sum')
    elif loss_fnc == "huber":
        loss_fnc = nn.HuberLoss(reduction='sum', delta=2.0)
    elif loss_fnc == "mse":
        loss_fnc = nn.MSELoss(reduction='sum')
    elif loss_fnc == "smooth_loss":
        loss_fnc = nn.SmoothL1Loss(reduction='sum')"""
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    train_model(model, optimizer, train_dl, batch_size, n_epochs)
    perc_err1, perc_err2 = eval_model(model, test_dl, batch_size)

    return (perc_err1, perc_err2)


def plottimeloss(datas):
    #datas[distance, true time, est. time]
    i1 = random.randint(0, len(datas)-1)
    i2 = random.randint(0, len(datas)-1)
    if i1 == i2:
        i2 += 1
    indis = np.concatenate([datas[i1], datas[i2]], axis=0)
    sorted_data = sorted(indis, key=lambda x: x[0])
    time_diffs = []
    for data in sorted_data:
        time_diff = mape(data[2],data[1])
        time_diffs.append(time_diff)
    dists = np.array([i[0] for i in sorted_data]).reshape(-1,1)
    time_diffs = np.array(time_diffs).reshape(-1,1)
    plt.plot(dists, time_diffs)
    plt.grid(True)
    plt.xlabel('Distance (m)')
    plt.ylabel('Percentage Time Difference [%Actual Time(s)]')
    plt.title('Percentage Time Difference vs. Distance')
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(f"Percentage Time Loss,{timestamp}")
    plt.close()


def plotloss(l1, l2, name):
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('X-axis')
    ax1.set_ylabel('loss', color=color)
    ax1.plot(l1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()  # shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('l2', color=color)  # we already handled the x-label with ax1
    ax2.plot(l2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()  # otherwise, the right y-label is slightly clipped
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(f"{name,timestamp}.png")
    plt.close(fig)

def normalize(data):
    """inputs are [:,:15]"""
    mean = data[:, :15].mean(dim=1, keepdim=True)
    std = data[:, :15].std(dim=1, keepdim=True)
    norm_data = (data[:, :15] - mean) / std
    mask = np.isnan(norm_data).bool()
    norm_data[mask] = 0
    return norm_data.to(torch.float64), mean, std

def denormalize(data, mean, std):
    denorm_data = data * std + mean
    return denorm_data.to(torch.float64)


def plot3dvectors(datas):
    fig = plt.figure(figsize=[15,10])
    ax = fig.add_subplot(111, projection='3d')
    plt.axis(xmin=-8, xmax=8, ymin =-4, ymax=4, zmin=-4, zmax=4)
    points = np.linspace(-8,8,104)
    i1 = random.randint(0, len(datas)-105)
    for i, data in enumerate(datas[i1:i1+104]):
        ind = random.randint(0, data.shape[0]-1)
        # Plot the first set of vectors in red
        #div. by 5 to scale max 20 to 4
        true_vec = np.array((data[ind][3], data[ind][4], data[ind][5]))
        est_vec = np.array((data[ind][6], data[ind][7], data[ind][8]))
        ax.quiver(points[i], 0, 0, *true_vec*0.3, color='g', linestyle='dashed')
        # Plot the second set of vectors in blue
        ax.quiver(points[i], 0, 0, *est_vec*0.3, color='r', linestyle='dashed')
        i += 3

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(f"vectorscomparison,{timestamp}.png")
    plt.close()




def train_model(model, optimizer, train_dl, batch_size, n_epochs):
    errs1 = []
    errs2 = []
    model.train(True)
    sample_count = 0
    loss_fnc1 = nn.MSELoss().requires_grad_(True)
    loss_fnc2 = nn.SmoothL1Loss().requires_grad_(True)

    for epoch in range(n_epochs+1):
        run_l1 = 0.0
        run_l2 = 0.0
        for batch in train_dl:
            make_dot(model(batch[:, :15]), params=dict(list(model.named_parameters()))).render("mymodel_torchviz",
                                                                                               format='png')
            print("Rendering is finished")
            if len(batch) < batch_size:
                continue
            batch[:, :15], mean, std = normalize(batch)
            inputs = batch[:, :15].clone().detach().requires_grad_(True).to(DEVICE).to(torch.float64)
            targets1 = batch[:, 15:18].clone().detach().requires_grad_(True).to(DEVICE).to(torch.float64)
            targets2 = batch[:, -1].clone().detach().requires_grad_(True).to(DEVICE).to(torch.float64)
            targets2 = targets2.reshape(-1, 1)
            pose_outs, time_outs = model(inputs)
            time_outs = denormalize(time_outs.reshape(-1, 1), mean, std)
            pose_outs = denormalize(pose_outs, mean, std)
            assert pose_outs.shape == targets1.shape
            loss1 = loss_fnc1(pose_outs, targets1)
            assert time_outs.shape == targets2.shape
            loss2 = loss_fnc2(time_outs, targets2)
            sample_count += batch_size
            run_l1 += loss1.item()
            run_l2 += loss2.item()
            loss = loss1 + loss2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if sample_count % 999 == 0:
                print(f'\t\t\t Target \t Output  \n')
                for target1, pose_out, target2, time_out in zip(targets1.data.numpy(),
                        pose_outs.data.numpy(),targets2.data.numpy(), time_outs.data.numpy()):
                    print(f'{target1} | {pose_out} | \n| {target2} | {time_out} ')
                print("\t Loss:", loss1.item(), loss2.item())

                #for name, param in model.named_parameters():
                    #print(name, param.grad.abs().sum())

        errs1.append(run_l1/len(train_dl))
        errs2.append(run_l1/len(train_dl))

    plotloss(errs1, errs2, 'TrainingLoss')

def mape(outs, targets):
    if torch.is_tensor(targets):
        if torch.norm(targets) >= torch.norm(outs):
            mask = targets == 0
            targets[mask] = 0.01
            abs_perc_error = torch.abs((targets - outs)/targets)*100
            mape = torch.mean(abs_perc_error)
        else:
            mask = outs == 0
            outs[mask] = 0.01
            abs_perc_error = torch.abs((targets - outs) / outs) * 100
            mape = torch.mean(abs_perc_error)
        return mape
    if isinstance(outs, float):
        perc_err = abs((targets-outs)/max(targets, outs)) * 100
        return perc_err
    if isinstance(outs, array):
        if np.norm(targets) >= np.norm(outs):
            mask = targets == 0
            targets[mask] = 0.01
            abs_perc_error = np.abs((targets - outs) / targets) * 100
            mape = np.mean(abs_perc_error)
        else:
            mask = outs == 0
            outs[mask] = 0.01
            abs_perc_error = np.abs((targets - outs) / outs) * 100
            mape = np.mean(abs_perc_error)
        return mape


def eval_model(model, test_dl, batch_size):
    model.eval()
    perc_errs1 = []
    perc_errs2 = []
    with ((torch.no_grad())):

        vec_eval = []
        time_eval = []
        runloss1 = 0
        runloss2 = 0
        cnt = 0
        for batch in test_dl:

            if len(batch) < batch_size:
                continue
            gates_p = batch[:, 12:15]
            distances = torch.linalg.norm(batch[:, 6:9]-gates_p, dim=1) + torch.linalg.norm(gates_p-batch[:,:3], dim=1)
            batch[:, :15], mean, std = normalize(batch)
            input_tensors = batch[:, :15].clone().detach().to(DEVICE).to(torch.float64)
            targets1 = batch[:, 15:18].clone().detach().to(DEVICE).to(torch.float64)
            targets2 = batch[:, -1].clone().detach().to(DEVICE).to(torch.float64)
            targets2 = targets2.reshape(-1, 1)
            pose_outs, time_outs = model(input_tensors)
            time_outs = denormalize(time_outs.reshape(-1, 1), mean, std)
            pose_outs = denormalize(pose_outs, mean, std)
            vectoappend = torch.concatenate((gates_p, targets1, pose_outs), dim=1)
            vec_eval.append(vectoappend)
            timtoappend = torch.concatenate((distances.reshape(-1,1), targets2, time_outs), dim=1)
            time_eval.append(timtoappend)
            """#trying.......
            mydata = np.vstack([pose_out.detach().numpy(), time_out.detach().numpy()])
            pose_out, time_out = normalize_data(mydata, nrm_constants[i, :, 15:])"""
            #loss1 = torch.mean(((targets1 - pose_outs) ** 2) / batch_size)
            assert time_outs.shape == targets2.shape
            #loss2 = torch.abs((targets2 - time_outs)) / batch_size
            loss1 = mape(pose_outs, targets1)
            loss2 = mape(time_outs, targets2)
            runloss1 += loss1
            runloss2 += loss2
            # max_element1 = max(max_element1, torch.norm(target1))
            # max_element2 = max(max_element2, target2.item())
            cnt += batch_size
            perc_errs1.append(loss1/batch_size)
            perc_errs2.append(loss2/batch_size)
            if cnt % 499 == 0:
                print(f'\t\t\t Target \t Output  \n')
                for target1, pose_out, target2, time_out in zip(targets1.data.numpy(),
                                    pose_outs.data.numpy(), targets2.data.numpy(), time_outs.data.numpy()):
                    print(f'{target1} | {pose_out} | \n| {target2} | {time_out} ')
                print("\t Loss:", runloss1.item() / cnt, runloss2.item() / cnt)

        vec_eval = np.array(vec_eval)
        plot3dvectors(vec_eval)
        time_eval = np.array(time_eval)
        plottimeloss(time_eval)
        avg_perc_err1 = runloss1 / cnt
        avg_perc_err2 = runloss2 / cnt
        print("avg_perc_err1,avg_perc_err2 ",avg_perc_err1.item(), avg_perc_err2.item())
        plotloss(perc_errs1, perc_errs2, 'EvaluationLoss')
        # for the best model evaluation
        # print( "perc_errs1, perc_errs2 :: ", perc_errs1, perc_errs2)
        # return perc_errs1, perc_errs2
        perc_error1 = np.mean(np.array(perc_errs1))
        perc_error2 = np.mean(np.array(perc_errs2))
        print("percentage err in vel. vector, err in time:", perc_error1, perc_error2)
        return perc_error1, perc_error2


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
    """
    #to search for a new model with optuna 
    study = optuna.create_study(directions=["minimize", "minimize"])
    study.optimize(objective, n_trials=25)
    print("Number of finished trials: ", len(study.trials))
    print(f"Number of trials on the front: {len(study.best_trials)}")
    print(study.get_trials())
    print(study.best_trials)
    trial_with_highest_accuracy = min(study.best_trials, key=lambda t: t.values)
    print(f"Trial with highest accuracy: ")
    print(f"\tnumber: {trial_with_highest_accuracy.number}")
    print(f"\tparams: {trial_with_highest_accuracy.params}")
    print(f"\tvalues: {trial_with_highest_accuracy.values}")
    fig = optuna.visualization.plot_param_importances(study)
    fig.show()
    fig1 = plot_optimization_history(study, target=lambda t: t.values[0], target_name='perc_err1')
    fig1.show()
    fig2 = plot_optimization_history(study, target=lambda t: t.values[1], target_name='perc_err2')
    fig2.show()
    # Loss Function
    """
    # Initialize Model with the best parameters:
    params = {'batch_size': 40, 'num_layers': 45, 'num_epochs': 1, 'learning_rate': 6.38896244782989e-05,
             'n_units_l0': 33, 'n_units_l1': 83, 'n_units_l2': 57, 'n_units_l3': 230, 'n_units_l4': 73,
             'n_units_l5': 46, 'n_units_l6': 89, 'n_units_l7': 256, 'n_units_l8': 140, 'n_units_l9': 98,
             'n_units_l10': 81, 'n_units_l11': 133, 'n_units_l12': 51, 'n_units_l13': 172, 'n_units_l14': 60,
             'n_units_l15': 64, 'n_units_l16': 166, 'n_units_l17': 139, 'n_units_l18': 81, 'n_units_l19': 182,
             'n_units_l20': 53, 'n_units_l21': 68, 'n_units_l22': 193, 'n_units_l23': 158, 'n_units_l24': 64,
             'n_units_l25': 44, 'n_units_l26': 35, 'n_units_l27': 100, 'n_units_l28': 181, 'n_units_l29': 243,
             'n_units_l30': 87, 'n_units_l31': 36, 'n_units_l32': 122, 'n_units_l33': 230, 'n_units_l34': 141,
             'n_units_l35': 144, 'n_units_l36': 93, 'n_units_l37': 82, 'n_units_l38': 102, 'n_units_l39': 65,
             'n_units_l40': 167, 'n_units_l41': 35, 'n_units_l42': 87, 'n_units_l43': 49, 'n_units_l44': 144}
    learning_rate = params["learning_rate"]
    batch_size = params["batch_size"]
    n_epochs = params["num_epochs"]
    all_data = TrajectoryOptimizationDataset("samples2")
    train_data, test_data = train_test_split(all_data, test_size=0.2, random_state=42)
    in_size = len(train_data[0]) - 4
    out_size = 4
    model = create_bestmodel(in_size, out_size, params).to(DEVICE)
    train_dl = DataLoader(train_data, batch_size, shuffle=True)
    test_dl = DataLoader(test_data, batch_size, shuffle=False)
    loss_fnc = "mse"
    if loss_fnc == "l1":
        loss_fnc = nn.L1Loss(reduction='sum')
    elif loss_fnc == "huber":
        loss_fnc = nn.HuberLoss(reduction='sum', delta=2.0)
    elif loss_fnc == "mse":
        loss_fnc = nn.MSELoss(reduction='sum')
    elif loss_fnc == "smooth_loss":
        loss_fnc = nn.SmoothL1Loss(reduction='sum')
    elif loss_fnc == "MAPE":
        loss_fnc = mape

    model.load_state_dict(torch.load("time_and_vel_vector.pth"))

    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    # Train Model
    train_model(model, optimizer, train_dl, batch_size, n_epochs)
    # Evaluate Model
    perc_err1, perc_err2 = eval_model(model, test_dl, batch_size)
    print("perc_err1, perc_err2: ", perc_err1, perc_err2)
    torch.save(model.state_dict(), "time_and_vel_vector.pth")
    print("saved.")

