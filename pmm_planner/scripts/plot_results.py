"""
    _summary_ Visualization tools for PMM multiwaypoint trajectories
    _author__ = "Krystof Teissing"
    _email_ = "k.teissing@gmail.com"

"""

import csv
import string
from traceback import print_tb
import numpy as np
import pandas as pd
from csv import reader
import matplotlib.pyplot as plt
import matplotlib.collections as mc

import plot_pmm

class pmm_trajectory:
    def __init__(self) -> None:
        self.n_segments = 0
        self.t = np.empty(0)
        self.p = np.empty(0)
        self.v = np.empty(0)
        self.a = np.empty(0)
        self.dt = np.empty(0)
        self.dtdv = np.empty(0)

    def load_trajectory_data_from_csv(self, file_name: string):
        with open(file_name, 'r') as file:
            csv_reader = reader(file)
            csv_data = [row for row in csv_reader]
            self.n_segments = csv_data[0][0]
            self.t = np.array(csv_data[1]).astype(np.double)
            self.p = np.array(csv_data[2]).astype(np.double)
            self.v = np.array(csv_data[3]).astype(np.double)
            self.a = np.array(csv_data[4]).astype(np.double)
            self.dt = np.array(csv_data[5]).astype(np.double)
            self.dtdv = np.array(csv_data[6]).astype(np.double)


    def plot_trajectory_data(self, color: str, axs, legend: string):
        if self.n_segments == 0:
            print("No trajectory data to plot\n")
            return None

        if axs is None:
            # create new figure
            fig, axs = plt.subplots(3,1, figsize=(10,10))

        marker_size = 10
        line_width = 2

        # position
        axs[0].plot(self.t, self.p,".-", color= color, linewidth=line_width, markersize=marker_size, label=legend)
        axs[0].set_xlabel("t [s]")
        axs[0].set_ylabel("p [m]")
        axs[0].grid(True)
        axs[0].legend()

        # velocity
        axs[1].plot(self.t, self.v, ".-", color= color, linewidth=line_width, markersize=marker_size)
        axs[1].set_xlabel("t [s]")
        axs[1].set_ylabel("v [m/s]")
        axs[1].grid(True)

        # acceleration
        a = np.repeat(self.a, 2)
        t = np.repeat(self.t,2)[1:-1]
        a_segments = [[(t[2*i], a[2*i]), (t[(2*i)+1], a[(2*i)+1])] for i in range(self.t.shape[0]-1)]
        lc = mc.LineCollection(a_segments, color= color ,linewidth=line_width)
        axs[2].add_collection(lc)
        axs[2].scatter(t, a, s=marker_size, c=color)
        axs[2].set_xlabel("t [s]")
        axs[2].set_ylabel("a [m/s^2]")
        axs[2].grid(True)
        return axs

class pmm_trajectory_3d:
    def __init__(self) -> None:
        self.n_segments = 0
        self.t = np.empty(0)
        self.p = np.empty((3,0))
        self.v = np.empty((3,0))
        self.a = np.empty((3,0))
        self.dt = np.empty(0)
        self.dtdv = np.empty((3,0))
        self.used_grad = np.empty((3,0))

    def load_trajectory_data_from_csv(self, file_name: string):
        with open(file_name, 'r') as file:
            csv_reader = reader(file)
            
            try:
                nl = next(csv_reader)
            except:
                print("Error")
            self.n_segments = int(nl[0])
            # init arrays to correct size
            self.p = np.empty((3,3 * self.n_segments +1))
            self.t = np.empty((3,3 * self.n_segments +1))
            self.v = np.empty((3,3 * self.n_segments +1))
            self.a = np.empty((3,3 * self.n_segments))
            self.dtdv = np.empty((3,2 * self.n_segments))
            self.dt = np.empty((3,3 * self.n_segments))

            for i in range(3):
                self.p[i,:] = np.array(next(csv_reader)).astype(np.double)
            for i in range(3):
                self.v[i,:] = np.array(next(csv_reader)).astype(np.double)
            for i in range(3):
                self.a[i,:] = np.array(next(csv_reader)).astype(np.double)
            for i in range(3):
                self.dtdv[i,:] = np.array(next(csv_reader)).astype(np.double)
            for i in range(3):
                self.dt[i,:] = np.array(next(csv_reader)).astype(np.double)
        
        # compute t
        t_integrator = np.array([0.0,0.0,0.0])
        self.t[:,0] = t_integrator
        for i in range(3*self.n_segments):
            t_integrator += self.dt[:,i]
            self.t[:,i+1] = t_integrator


    def plot_trajectory_data(self, color: list, axs, legend: list, title:string):
        if self.n_segments == 0:
            print("No trajectory data to plot\n")
            return None

        if axs is None:
            # create new figure
            fig, axs = plt.subplots(3,1, figsize=(10,10))
            fig.suptitle(title, fontsize=16)

        marker_size = 10
        line_width = 2

        # position
        axs[0].plot(self.t[0,:], self.p[0,:],".-", color= color[0], linewidth=line_width, markersize=marker_size, label=legend[0])
        axs[0].plot(self.t[1,:], self.p[1,:],".-", color= color[1], linewidth=line_width, markersize=marker_size, label=legend[1])
        axs[0].plot(self.t[2,:], self.p[2,:],".-", color= color[2], linewidth=line_width, markersize=marker_size, label=legend[2])
        axs[0].set_xlabel("t [s]")
        axs[0].set_ylabel("p [m]")
        axs[0].grid(True)
        axs[0].legend()

        # velocity
        axs[1].plot(self.t[0,:], self.v[0,:], ".-", color= color[0], linewidth=line_width, markersize=marker_size)
        axs[1].plot(self.t[1,:], self.v[1,:], ".-", color= color[1],linewidth=line_width, markersize=marker_size)
        axs[1].plot(self.t[2,:], self.v[2,:], ".-", color= color[2], linewidth=line_width, markersize=marker_size)
        axs[1].set_xlabel("t [s]")
        axs[1].set_ylabel("v [m/s]")
        axs[1].grid(True)

        # acceleration
        for i in range(3):
            a = np.repeat(self.a[i,:], 2)
            t = np.repeat(self.t[i,:],2)[1:-1]
            a_segments = [[(t[2*i], a[2*i]), (t[(2*i)+1], a[(2*i)+1])] for i in range(self.t.shape[1]-1)]
            lc = mc.LineCollection(a_segments, color= color[i] ,linewidth=line_width)
            axs[2].add_collection(lc)
            axs[2].scatter(t, a, s=marker_size, c=color[i])
        axs[2].set_xlabel("t [s]")
        axs[2].set_ylabel("a [m/s^2]")
        axs[2].grid(True)
        return axs
    
class pmm_sampled_trajectory_3d:
    def __init__(self) -> None:
        self.t = np.empty(0)
        self.p = np.empty((3,0))
        self.v = np.empty((3,0))
        self.a = np.empty((3,0))

    def load_trajectory_data_from_csv(self, file_name: string):
        with open(file_name, 'r') as file:
            csv_reader = reader(file)
            
            try:
                nl = next(csv_reader)
            except:
                print("Error")
            self.t = np.array(nl).astype(np.double)
            n_wp = self.t.shape[0]
            # init arrays to correct size
            self.p = np.empty((3,n_wp))
            self.v = np.empty((3,n_wp))
            self.a = np.empty((3,n_wp))

            for i in range(3):
                self.p[i,:] = np.array(next(csv_reader)).astype(np.double)
            for i in range(3):
                self.v[i,:] = np.array(next(csv_reader)).astype(np.double)
            for i in range(3):
                self.a[i,:] = np.array(next(csv_reader)).astype(np.double)

    def plot_trajectory_data(self, color: list, axs, legend: list, title:string):
        if axs is None:
            # create new figure
            fig, axs = plt.subplots(3,1, figsize=(10,10))
            fig.suptitle(title, fontsize=16)

        marker_size = 2
        line_width = 2

        # position
        axs[0].plot(self.t, self.p[0,:],".-", color= color[0], linewidth=line_width, markersize=marker_size, label=legend[0])
        axs[0].plot(self.t, self.p[1,:],".-", color= color[1], linewidth=line_width, markersize=marker_size, label=legend[1])
        axs[0].plot(self.t, self.p[2,:],".-", color= color[2], linewidth=line_width, markersize=marker_size, label=legend[2])
        axs[0].set_xlabel("t [s]")
        axs[0].set_ylabel("p [m]")
        axs[0].grid(True)
        axs[0].legend()

        # velocity
        axs[1].plot(self.t, self.v[0,:], ".-", color= color[0], linewidth=line_width, markersize=marker_size)
        axs[1].plot(self.t, self.v[1,:], ".-", color= color[1],linewidth=line_width, markersize=marker_size)
        axs[1].plot(self.t, self.v[2,:], ".-", color= color[2], linewidth=line_width, markersize=marker_size)
        axs[1].set_xlabel("t [s]")
        axs[1].set_ylabel("v [m/s]")
        axs[1].grid(True)

        # acceleration
        axs[2].plot(self.t, self.a[0,:], ".-", color= color[0], linewidth=line_width, markersize=marker_size)
        axs[2].plot(self.t, self.a[1,:], ".-", color= color[1],linewidth=line_width, markersize=marker_size)
        axs[2].plot(self.t, self.a[2,:], ".-", color= color[2], linewidth=line_width, markersize=marker_size)
        axs[2].set_xlabel("t [s]")
        axs[2].set_ylabel("a [m/s^2]")
        axs[2].grid(True)
        return axs
    
class pmm_sampled_path_3d:
    def __init__(self) -> None:
        self.t = np.empty(0)
        self.p = np.empty((3,0))

    def load_trajectory_data_from_csv(self, file_name: string):
        with open(file_name, 'r') as file:
            csv_reader = reader(file)
            
            try:
                nl = next(csv_reader)
            except:
                print("Error")
            self.t = np.array(nl).astype(np.double)
            n_wp = self.t.shape[0]
            # init arrays to correct size
            self.p = np.empty((3,n_wp))

            for i in range(3):
                self.p[i,:] = np.array(next(csv_reader)).astype(np.double)

    def plot_trajectory_data(self, color: list, axs, legend: list, title:string):
        if axs is None:
            # create new figure
            fig, axs = plt.subplots(1,1, figsize=(10,10))
            fig.suptitle(title, fontsize=16)

        marker_size = 10
        line_width = 2

        # position
        axs.plot(self.t, self.p[0,:],".-", color= color[0], linewidth=line_width, markersize=marker_size, label=legend[0])
        axs.plot(self.t, self.p[1,:],".-", color= color[1], linewidth=line_width, markersize=marker_size, label=legend[1])
        axs.plot(self.t, self.p[2,:],".-", color= color[2], linewidth=line_width, markersize=marker_size, label=legend[2])
        axs.set_xlabel("t [s]")
        axs.set_ylabel("p [m]")
        axs.grid(True)
        axs.legend()
        return axs

class pmm_3D_GD_data:
    def __init__(self, file_name: string) -> None:
        self.marker_size = 10

        with open(file_name, 'r') as file:
            csv_reader = reader(file)
            # self.n_iterations = int(list(csv_reader)[-1][0])
            self.data = []

            while True:
                try:
                    nl = next(csv_reader)
                except:
                    break
                tr_tmp = pmm_trajectory_3d()
                tr_tmp.n_segments = int(nl[0])
                # init arrays to correct size
                tr_tmp.p = np.empty((3,3 * tr_tmp.n_segments +1))
                tr_tmp.t = np.empty((3,3 * tr_tmp.n_segments +1))
                tr_tmp.v = np.empty((3,3 * tr_tmp.n_segments +1))
                tr_tmp.a = np.empty((3,3 * tr_tmp.n_segments))
                tr_tmp.dt = np.empty((3,3 * tr_tmp.n_segments))
                tr_tmp.dtdv = np.empty((3,2 * tr_tmp.n_segments))
                tr_tmp.used_grad = np.empty((3, tr_tmp.n_segments))

                for i in range(3):
                    tr_tmp.p[i,:] = np.array(next(csv_reader)).astype(np.double)
                for i in range(3):
                    tr_tmp.v[i,:] = np.array(next(csv_reader)).astype(np.double)
                for i in range(3):
                    tr_tmp.a[i,:] = np.array(next(csv_reader)).astype(np.double)
                for i in range(3):
                    tr_tmp.dtdv[i,:] = np.array(next(csv_reader)).astype(np.double)
                for i in range(3):
                    tr_tmp.dt[i,:] = np.array(next(csv_reader)).astype(np.double)
                for i in range(3):
                    tr_tmp.used_grad[i,:] = np.array(next(csv_reader)).astype(np.double)
                # compute t
                t_integrator = np.array([0.0,0.0,0.0])
                tr_tmp.t[:,0] = t_integrator
                for i in range(3*tr_tmp.n_segments):
                    t_integrator += tr_tmp.dt[:,i]
                    tr_tmp.t[:,i+1] = t_integrator
                self.data.append(tr_tmp)
            self.n_iterations = len(self.data)
    
    def plot_tr_times(self, axs, color):
        T = np.array([np.sum(tr.dt[0,:]) for tr in self.data])
        axs.plot(np.arange(self.n_iterations), T, ".-", color=color, markersize=self.marker_size)
        axs.set_xlabel("iter [-]")
        axs.set_ylabel("T [s]")
        axs.grid(True)

    def plot_tr_times_diff(self, axs, color):
        T = np.array([np.sum(tr.dt[0,:]) for tr in self.data])
        dT = np.ediff1d(T)
        axs.plot(np.arange(self.n_iterations-1), dT, ".-", color=color, markersize=self.marker_size)
        axs.set_xlabel("iter [-]")
        axs.set_ylabel("dT [s]")
        axs.grid(True)

    def plot_dvdt(self, segment_idx, axs, color:list, legend:list):
        dtdv = np.array([tr.dtdv[:, (2*segment_idx) + 1] + tr.dtdv[:, (2*segment_idx) + 2] for tr in self.data])
        axs.plot(np.arange(self.n_iterations), dtdv[:,0],".-", color=color[0], markersize=self.marker_size, label=legend[0])
        axs.plot(np.arange(self.n_iterations), dtdv[:,1],".-", color=color[1], markersize=self.marker_size, label=legend[1])
        axs.plot(np.arange(self.n_iterations), dtdv[:,2],".-", color=color[2], markersize=self.marker_size, label=legend[2])
        axs.set_xlabel("iter [-]")
        axs.set_ylabel("dt/dv [-]")
        axs.grid(True)
        axs.legend()

    def plot_used_grad(self, segment_idx, axs, color:list, legend:list):
        grad = np.array([tr.used_grad[:,segment_idx] for tr in self.data])
        axs.plot(np.arange(self.n_iterations), grad[:,0],".-", color=color[0], markersize=self.marker_size, label=legend[0])
        axs.plot(np.arange(self.n_iterations), grad[:,1],".-", color=color[1], markersize=self.marker_size, label=legend[1])
        axs.plot(np.arange(self.n_iterations), grad[:,2],".-", color=color[2], markersize=self.marker_size, label=legend[2])
        axs.set_xlabel("iter [-]")
        axs.set_ylabel("dt/dv [-]")
        axs.grid(True)
        axs.legend()

    def plot_v(self, v_idx, axs, color:list, legend: list):
        v = np.array([tr.v[:, v_idx] for tr in self.data])
        axs.plot(np.arange(self.n_iterations), v[:,0],".-", color=color[0], markersize=self.marker_size, label=legend[0])
        axs.plot(np.arange(self.n_iterations), v[:,1],".-", color=color[1], markersize=self.marker_size, label=legend[1])
        axs.plot(np.arange(self.n_iterations), v[:,2],".-", color=color[2], markersize=self.marker_size, label=legend[2])
        axs.set_xlabel("iter [-]")
        axs.set_ylabel("v [m/s]")
        axs.grid(True)
        axs.legend()


def visualize_pmm_mp_3d():
    # file names
    orig_tr_file = "scripts/trajectory_data/dp_example_mp_tr_orig.csv"
    optim_tr_file = "scripts/trajectory_data/dp_example_mp_tr_optim.csv"
    optim_tr2_file = "scripts/trajectory_data/dp_example_mp_tr_optim_2nd_run.csv"
    gd_data_file = "scripts/trajectory_data/dp_example_mp_tr_data.csv"
    gd_data2_file = "scripts/trajectory_data/dp_example_mp_tr_data_2nd_run.csv"

    # plot original and optimized trajectory
    tr_orig = pmm_trajectory_3d()
    tr_orig.load_trajectory_data_from_csv(orig_tr_file)
    tr_optim = pmm_trajectory_3d()
    tr_optim.load_trajectory_data_from_csv(optim_tr_file)
    tr_optim2 = pmm_trajectory_3d()
    tr_optim2.load_trajectory_data_from_csv(optim_tr2_file)

    axs = tr_orig.plot_trajectory_data(['r', 'g', 'b'], None, ['x', 'y', 'z'], 'Original Trajectory')
    axs = tr_optim.plot_trajectory_data(['r', 'g', 'b'], None, ['x', 'y', 'z'], 'Optimized Trajectory After 1st Run')
    axs = tr_optim2.plot_trajectory_data(['r', 'g', 'b'], None, ['x', 'y', 'z'], 'Optimized Trajectory After 2nd Run')

    # plot GD data
    gd_data1 = pmm_3D_GD_data(gd_data_file)
    fig, axs1 = plt.subplots(2,1, figsize=(10,10))
    fig.suptitle('GD optimization data 1st run', fontsize=16)
    gd_data1.plot_tr_times(axs1[0], 'b')
    gd_data1.plot_tr_times_diff(axs1[1], 'b')

    # plot GD data
    gd_data2 = pmm_3D_GD_data(gd_data2_file)
    fig, axs2 = plt.subplots(2,1, figsize=(10,10))
    fig.suptitle('GD optimization data 2nd run', fontsize=16)
    gd_data2.plot_tr_times(axs2[0], 'b')
    gd_data2.plot_tr_times_diff(axs2[1], 'b')

def visualize_sampled_pmm_mp_3d():
    # file names
    orig_tr_file = "scripts/trajectory_data/dp_example_mp_tr_orig_sampl.csv"
    optim_tr_file = "scripts/trajectory_data/dp_example_mp_tr_optim_sampl.csv"
    optim_tr2_file = "scripts/trajectory_data/dp_example_mp_tr_optim_sampl_2nd_run.csv"

    # plot original and optimized trajectory
    tr_orig = pmm_sampled_trajectory_3d()
    tr_orig.load_trajectory_data_from_csv(orig_tr_file)
    tr_optim = pmm_sampled_trajectory_3d()
    tr_optim.load_trajectory_data_from_csv(optim_tr_file)
    tr_optim2 = pmm_sampled_trajectory_3d()
    tr_optim2.load_trajectory_data_from_csv(optim_tr2_file)

    axs = tr_orig.plot_trajectory_data(['r', 'g', 'b'], None, ['x', 'y', 'z'], 'Sampled Original Trajectory')
    axs = tr_optim.plot_trajectory_data(['r', 'g', 'b'], None, ['x', 'y', 'z'], 'Sampled Optimized Trajectory After 1st Run')
    axs = tr_optim2.plot_trajectory_data(['r', 'g', 'b'], None, ['x', 'y', 'z'], 'Sampled Optimized Trajectory After 2nd Run')

def visualize_cftg():
    # file names
    orig_tr_file = "scripts/trajectory_data/dp_example_cftg_sampl.csv"

    # plot original and optimized trajectory
    tr_orig = pmm_sampled_trajectory_3d()
    tr_orig.load_trajectory_data_from_csv(orig_tr_file)

    axs = tr_orig.plot_trajectory_data(['r', 'g', 'b'], None, ['x', 'y', 'z'], 'Collision-Free Trajectory')

def visualize_sampled_pmm_mp_3d_path():
    # file names
    tr_file = "scripts/trajectory_data/mp_tr_3D_sampled_path.CSV"

    # plot original and optimized trajectory
    tr_orig = pmm_sampled_path_3d()
    tr_orig.load_trajectory_data_from_csv(tr_file)

    axs = tr_orig.plot_trajectory_data(['r', 'g', 'b'], None, ['x', 'y', 'z'], 'Sampled Trajectory')


if __name__=="__main__":
    # print("Visualizing PMM Trajectory Data")
    # visualize_pmm_mp_3d()
    print("Visualizing Sampled Trajectories")
    visualize_sampled_pmm_mp_3d()
    # print("Visualizing Final Collision-Free Trajectory generated with CFTG")
    # visualize_cftg()

    tr_samples = plot_pmm.load_trajectory_samples_pmm("scripts/trajectory_data/dp_example_mp_tr_optim_sampl_2nd_run.csv")
    plot_pmm.plot_3d_positions_graph(tr_samples, {}, 'conf/vo_path_2.yaml')

    plt.show()