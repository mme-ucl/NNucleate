import torch
import numpy as np

def e_path_loss(model, calc, config, edges, edges_e, n_nodes, box, batch_size, beta, alpha=0.9, emax=10, L=5, device='cpu'):
    err = torch.zeros(batch_size).to(device)
    pred = model(x=config, edges=edges, n_nodes=n_nodes)
    epots = calc.calculate(config*10, edges_e, box*10)
    #gs0 = torch.autograd.grad(pred[:,0], config, grad_outputs=torch.ones_like(pred[:,0]), retain_graph=True)[0]
    gs = torch.autograd.grad(pred[:,1], config, grad_outputs=torch.ones_like(pred[:,1]), retain_graph=True)[0]
    #gs = gs0 + gs1
    #gs = torch.autograd.grad(pred, config, grad_outputs=torch.ones_like(pred), retain_graph=True)[0].to(device)
    gs = gs/np.max(gs.detach().cpu().numpy())*0.1

    for i in range(1,L):
        # move in the direction of the grads (- for larger pred towards 1)
        config = config + gs*1
        # reevaluate the prediction
        pred_new = model(x=config, edges=edges, n_nodes=n_nodes)
        # reevaluate the energies
        epots_new = calc.calculate(config*10, edges_e, box*10)
        delta_pred = torch.abs(torch.norm(pred_new, p=1) - torch.norm(pred, p=1))
        delta_epot = torch.abs(epots_new - epots)
        # Add the new error decayed by distance mapped between 0 and 1 and then scaled to emax
        err += alpha**(i) * torch.minimum(torch.abs(torch.log(delta_pred)-beta*delta_epot), torch.tensor(emax))
        epots = epots_new
        pred = pred_new
        #gs0 = torch.autograd.grad(pred[:,0], config, grad_outputs=torch.ones_like(pred[:,0]), retain_graph=True)[0]
        gs = torch.autograd.grad(pred[:,1], config, grad_outputs=torch.ones_like(pred[:,1]), retain_graph=True)[0]
        #gs = gs0 + gs1
        #gs = torch.autograd.grad(pred_new, config, grad_outputs=torch.ones_like(pred_new), retain_graph=True)[0].to(device)
        gs = gs/np.max(gs.detach().cpu().numpy())*0.1

    return err


import ase 

from scipy.interpolate import InterpolatedUnivariateSpline as spline
class EAM_ase:
    def __init__(self, **kwargs):
        self.read_potential(kwargs['potential'])

    def read_potential(self, filename):
        """Reads a LAMMPS EAM file in alloy or adp format
        and creates the interpolation functions from the data
        """
        if isinstance(filename, str):
            with open(filename) as fd:
                self._read_potential(fd)
        else:
            fd = filename
            self._read_potential(fd)

    def _read_potential(self, fd):

        lines = fd.readlines()
        def lines_to_list(lines):
            """Make the data one long line so as not to care how its formatted
            """
            data = []
            for line in lines:
                data.extend(line.split())
            return data
        self.header = lines[:1]
        data = lines_to_list(lines[1:])
        # eam form is just like an alloy form for one element
        self.Nelements = 1
        self.Z = np.array([data[0]], dtype=int)
        self.mass = np.array([data[1]])
        self.a = np.array([data[2]])
        self.lattice = [data[3]]
        self.nrho = int(data[4])
        self.drho = float(data[5])
        self.nr = int(data[6])
        self.dr = float(data[7])
        self.cutoff = float(data[8])
        n = 9 + self.nrho
        self.embedded_data = np.array([np.float64(data[9:n])])
        self.rphi_data = np.zeros([self.Nelements, self.Nelements,
                                      self.nr])
        effective_charge = np.float64(data[n:n + self.nr])
        # convert effective charges to rphi according to
        # http://lammps.sandia.gov/doc/pair_eam.html
        self.rphi_data[0, 0] = ase.units.Bohr * ase.units.Hartree * (effective_charge**2)
        self.density_data = np.array(
               [np.float64(data[n + self.nr:n + 2 * self.nr])])
        self.r = np.arange(0, self.nr) * self.dr
        self.rho = np.arange(0, self.nrho) * self.drho
        # choose the set_splines method according to the type
        self.set_splines()
    def set_splines(self):
        # this section turns the file data into three functions (and
        # derivative functions) that define the potential
        self.embedded_energy = np.empty(self.Nelements, object)
        self.electron_density = np.empty(self.Nelements, object)
        for i in range(self.Nelements):
            self.embedded_energy[i] = spline(self.rho,
                                             self.embedded_data[i], k=3)
            self.electron_density[i] = spline(self.r,
                                              self.density_data[i], k=3)
        self.phi = np.empty([self.Nelements, self.Nelements], object)
        # ignore the first point of the phi data because it is forced
        # to go through zero due to the r*phi format in alloy and adp
        for i in range(self.Nelements):
            for j in range(i, self.Nelements):
                self.phi[i, j] = spline(
                    self.r[1:],
                    self.rphi_data[i, j][1:] / self.r[1:], k=3)
                
    def calculate(self, config, edges, box_l, device='cpu'):
        row, col = edges
        # All the distances
        drs = config[row] - config[col]
        inv_box = 1.0 / box_l
        dr = drs - box_l * torch.round(drs * inv_box)

        rs = torch.norm(dr, dim=1)
        # Look up rhos
        rho = torch.Tensor(self.electron_density[0](rs.detach().cpu().numpy())).to(device)
        #print(torch.max(rs))
        # Look up phis
        phi = torch.Tensor(self.phi[0, 0](rs.detach().cpu().numpy())).to(device)

        rhos_pa = torch.zeros(config.size(0), dtype=torch.float32).to(device).scatter_add(0, row, rho)
        phis_pa = torch.zeros(config.size(0), dtype=torch.float32).to(device).scatter_add(0, row, phi)
        # Calculate the energy
        Eis = torch.tensor(self.embedded_energy[0](rhos_pa.detach().cpu().numpy())).to(device) + 0.5*phis_pa

        E = torch.zeros(int(config.size(0)/500), dtype=torch.float64).to(device).scatter_add(0,torch.tensor(np.repeat(np.arange(int(config.size(0)/500)), 500)).to(device), Eis)
        return E