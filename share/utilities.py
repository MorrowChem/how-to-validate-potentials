import os
import sys
from datetime import datetime
from os.path import join
import subprocess
from multiprocessing import Pool
import numpy as np
from scipy.stats import gaussian_kde
import pandas as pd
from ase.optimize import BFGSLineSearch
from ase.constraints import ExpCellFilter, FixAtoms
from ase.io import read, write
from ase.io.castep import read_castep_cell, write_castep_cell
from ase.atoms import Atoms
from warnings import warn


def read_lammps_log(file):
    '''Reads LAMMPS log thermo output into a pandas DataFrame

    Params: (str) filename
    Returns: (pandas DataFrame) table of thermo output data'''

    with open(file, 'r') as f:
        out = f.readlines()
    flag = False
    first_time = True
    for i, val in enumerate(out):
        test = val.split()
        test.append('')
        if test[0] == 'Step':
            if first_time:
                dat = [[] for j in range(len(out[i+1].split()))]
                dat_head = val.split()
                first_time = False
            flag = True
            continue

        if flag:
            try:
                for j, num in enumerate(val.split()):
                    dat[j].append(float(num))
            except:
                flag = 0

    dat = np.array(dat) 
    df = pd.DataFrame(dat[:].T, columns=dat_head[:]) # turn into a DataFrame with header
    df.drop_duplicates(subset=df.columns[-1], inplace=True)
    
    return df


def optimise_structure(atoms, potential, P=0,
                      steps=100, fmax=5e-2, traj='/dev/null',
                      silent=False, check_convergence=True, return_opt=False):
    '''Optimise structure using BFGS at a given pressure
    Params: atoms (ase.Atoms) structure to optimise
            potential (ase.calculator) potential to use
            P (float) pressure to apply in eV/A^3
            steps (int) max steps for optimisation
            fmax (float) max force criterion for stopping optimisation
            traj (str) filename for trajectory
            silent (bool) silence optimisation output
            check_convergence (bool) warn if not converged
            return_opt (bool) return optimiser object or not
    Returns: (ase.Atoms) atoms'''


    atoms.calc = potential
    atoms.set_constraint(FixAtoms(mask=[True for atom in atoms])) # only optimise lattice
    uf = ExpCellFilter(atoms, scalar_pressure=P, hydrostatic_strain=True) # should ensure only the lattice can move, not atomic positions


    if silent:
        opt = BFGSLineSearch(atoms=uf, trajectory=traj, logfile='/dev/null')
    else:
        opt = BFGSLineSearch(atoms=uf, trajectory=traj)

    opt.run(fmax, steps=steps)

    if check_convergence and not opt.converged():
        warn(('Warning: failed to converge on structure\n' +\
                        'in {} steps').format(steps))
    atoms.calc = None
    
    if return_opt:
        return atoms, opt
    else:
        return atoms


def despine(ax):
    '''Removes all spines from matplotlib axes'''
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)


def density_estimation(m1, m2, xmin, xmax, ymin, ymax, bw=None, point_density=100j):
    """esimates gaussian density of two variables for countour plotting

    # Args:
        m1 (array): x data
        m2 (array): y data
        xmin (float): smallest x value for kernel evaluation. Works best if slightly smaller than actual min(x)
        xmax (float): largest x value for kernel evaluation. Works best if slightly larger than actual max(x)
        ymin (float): 
        ymax (float): 
        bw (float, optional): bin width for Gaussian. Defaults to scipy value of 0.1.
        point_density (complex, optional): density of grid for Gaussian evaluation. Defaults to 100j.

    # Returns:
        tuple: X, Y, Z data for contour plotting
    """

    X, Y = np.mgrid[xmin:xmax:point_density, ymin:ymax:point_density]                                                     
    positions = np.vstack([X.ravel(), Y.ravel()])                                                       
    values = np.vstack([m1, m2])                                                                        
    kernel = gaussian_kde(values, bw_method=bw)                                                                 
    Z = np.reshape(kernel(positions).T, X.shape)

    return X, Y, Z


def update_progress(progress):
    barLength = 30 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1:5.2f}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()


def run_buildcell(arg):
    '''Helper function to run buildcellat top-level (multiprocessing)
    Parameters:
        arg :: arguments for buildcell (list of strs)
    Returns:
        cell
    '''

    flag = True
    ctmax = 5
    ct = 0
    while flag:
        out = subprocess.run(arg, shell=True, capture_output=True,
                             text=True, timeout=30, check=True)
        try:
            cell = read_castep_cell(arg.split('>')[-1])
            flag = False
        except:
            ct += 1
            flag = True
            print('Failed {} times'.format(ct))
            if ct >= ctmax:
                cell = None
                return cell

    return cell


class RSS:
    '''A class to store and analyse a batch of structures generated using RSS'''


    def __init__(self, directory, buildcell_command='buildcell'):
        '''
        Parameters:
        directory - a directory in which to i/o structures
        buildcell_command (str, optional) path to your buildcell binary'''
        self.buildcell_command = buildcell_command
        self.directory = directory
        self.init_atoms = {}
        
        if not os.path.isdir(directory):
            os.mkdir(directory)

        return

    def buildcells(self, N, buildcell_options,
                   atoms=[], tag='rss', timeout=100,
                   nprocs=1, append=False, config_type=None,
                   fragment=False, fragment_atoms=None):
        '''Prepares random cells in self.directory
        Arguments:
        N :: (int) number of cells to make
        buildcell options :: (list of str) e.g. ['VARVOL=20']
        atoms :: (optional - ignored if species in buildcell options) 
                    list of tuples for atom and number to include,
                    e.g. [(Mo1,1), (S1,3)]
        tag   :: (str) prefix for all files
        timeout :: timeout for buildcell in seconds
        append :: whether or not to append structures
                  to the existing ML_RSS.init_atoms list'''
        
        bc_file = join(self.directory,'{}.cell'.format(tag))

        if 'SPECIES' not in ' '.join(buildcell_options):
            if fragment:
                at = fragment_atoms

            else:
                at = Atoms(symbols=''.join([i[0][:-1] for i in atoms]),
                      cell=np.identity(3)*2, pbc=True,
                       positions=np.zeros((len(atoms), 3))
                      )
    
        
        #### writing the master buildcell ########################
        
            write_castep_cell(bc_file, at, positions_frac=True)
            
            with open(bc_file, 'r') as f:
                contents = f.readlines()
            
            flag = 0; ct = 0
            if not fragment:
                for i, val in enumerate(contents):
                    if '%BLOCK POSITIONS_FRAC' in val:
                        flag=1
                    elif '%ENDBLOCK POSITIONS_FRAC' in val:
                        flag=0
                    elif flag==1:
                        contents[i] = val.strip('\n') + ' # {} % NUM={}\n'.format(atoms[ct][0], atoms[ct][1])
                        ct+=1
            else:
                for i, val in enumerate(contents):
                    if '%BLOCK POSITIONS_FRAC' in val:
                        flag=1
                    elif '%ENDBLOCK POSITIONS_FRAC' in val:
                        flag=0
                    elif flag==1:
                        contents[i] = val.strip('\n') + ' # {} % NUM=1\n'.format(fragment_atoms.arrays['fragment_id'][ct])
                        ct+=1
                
        else:
            with open(bc_file, 'w') as f:
                contents=['']
                
            
        contents.extend(['#' + i + '\n' for i in buildcell_options])
        
        with open(bc_file, 'w') as f:
            f.writelines(contents)
        
        print('Building cells...')
        if nprocs == 1:  # unparallelised version
            startTime = datetime.now()
            for i in range(N):
                buildcell_command = [self.buildcell_command, '<', bc_file,
                                 '>', join(self.directory, '{}_{}.cell'.format(tag, i))] 
                buildcell_command = ''.join(buildcell_command)

                out = subprocess.run(buildcell_command, shell=True,
                                     capture_output=True, text=True,
                                     timeout=timeout, check=True)
     
                update_progress((i+1)/N)
            print('Walltime: ', datetime.now() - startTime)
            print('Reading cells...', end='')
            if append:
                self.init_atoms[tag].extend([read(
                    join(self.directory, '{}_{}.cell'.format(tag, i))) for i in range(N)])
            else:
                self.init_atoms[tag] = [read(
                    join(self.directory, '{}_{}.cell'.format(tag, i))) for i in range(N)]
        else:
            startTime = datetime.now()
            buildcell_commands = []

            for i in range(N):
                buildcell_command = [self.buildcell_command, '<', bc_file,
                                 '>', join(self.directory, '{}_{}.cell'.format(tag, i))] 
                buildcell_commands.append(''.join(buildcell_command))

            p = Pool(nprocs)
            fail = 0
            if not append:
                self.init_atoms[tag] = []
            for i, cell in enumerate(p.imap_unordered(run_buildcell, buildcell_commands, min(1, N//nprocs))):
                if cell is not None:
                    self.init_atoms[tag].append(cell)
                    if config_type is not None:
                        self.init_atoms[tag][-1].info['config_type'] = config_type
                else:
                    fail += 1
                    sys.stderr.write('failed cell count {}'.format(fail))
                sys.stderr.write('\rdone {:.2%}'.format((i+1)/N))
                sys.stderr.flush()
            p.close()
            p.join()
 
            print('\nWalltime: ', datetime.now() - startTime)

        for i in range(N):
            os.remove(join(self.directory, '{}_{}.cell'.format(tag, i)))

        # try writing all init_cells to file first, then access individually during
        # optimisation
        with open(join(self.directory, '{}.xyz'.format(tag)), 'w') as f:
            write(f, self.init_atoms[tag], append=append)