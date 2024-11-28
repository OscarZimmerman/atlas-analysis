import uproot
import awkward as ak
import numpy as np
import vector

def cut_photon_reconstruction(photon_isTightID):
    return (photon_isTightID[:, 0] == False) | (photon_isTightID[:, 1] == False)

def cut_photon_pt(photon_pt):
    return (photon_pt[:, 0] < 40) | (photon_pt[:, 1] < 30)

def cut_isolation_pt(photon_ptcone20):
    return (photon_ptcone20[:, 0] > 4) | (photon_ptcone20[:, 1] > 4)

def cut_photon_eta_transition(photon_eta):
    condition_0 = (np.abs(photon_eta[:, 0]) < 1.52) & (np.abs(photon_eta[:, 0]) > 1.37)
    condition_1 = (np.abs(photon_eta[:, 1]) < 1.52) & (np.abs(photon_eta[:, 1]) > 1.37)
    return condition_0 | condition_1

def calc_mass(photon_pt, photon_eta, photon_phi, photon_e):
    p4 = vector.zip({"pt": photon_pt, "eta": photon_eta, "phi": photon_phi, "e": photon_e})
    return (p4[:, 0] + p4[:, 1]).M

def process_tree(tree, variables, fraction=1.0):
    sample_data = []
    for data in tree.iterate(variables, library="ak", entry_stop=tree.num_entries * fraction):
        data = data[~cut_photon_reconstruction(data['photon_isTightID'])]
        data = data[~cut_photon_pt(data['photon_pt'])]
        data = data[~cut_isolation_pt(data['photon_ptcone20'])]
        data = data[~cut_photon_eta_transition(data['photon_eta'])]
        data['mass'] = calc_mass(data['photon_pt'], data['photon_eta'], data['photon_phi'], data['photon_e'])
        sample_data.append(data)
    return ak.concatenate(sample_data)
