import numpy as np
import matplotlib.pyplot as plt
import os
from lmfit.models import PolynomialModel, GaussianModel

def fit_data(bin_centres, data_x, data_x_errors):
    polynomial_mod = PolynomialModel(4)
    gaussian_mod = GaussianModel()
    
    pars = polynomial_mod.guess(data_x, x=bin_centres, c0=data_x.max())
    pars += gaussian_mod.guess(data_x, x=bin_centres, amplitude=100, center=125, sigma=2)
    
    model = polynomial_mod + gaussian_mod
    return model.fit(data_x, pars, x=bin_centres, weights=1 / data_x_errors)

import matplotlib.pyplot as plt
import os

def plot_results(bin_centres, data_x, data_x_errors, fit_result, background,
                 step_size, xmin, xmax, output_dir):
    """
    Plots the results and saves the plot to the specified output directory.
    """
    # Main Plot
    plt.figure(figsize=(10, 8))
    plt.axes([0.1, 0.3, 0.85, 0.65])  # left, bottom, width, height 
    main_axes = plt.gca()

    main_axes.errorbar(x=bin_centres, y=data_x, yerr=data_x_errors, fmt='ko', label='Data')
    main_axes.plot(bin_centres, fit_result.best_fit, '-r', label='Sig+Bkg Fit ($m_H=125$ GeV)')
    main_axes.plot(bin_centres, background, '--r', label='Bkg (4th order polynomial)')

    main_axes.set_xlim(left=xmin, right=xmax)
    main_axes.set_ylabel(f'Events / {step_size} GeV', horizontalalignment='right')
    main_axes.legend(frameon=False, loc='lower left')
    main_axes.tick_params(which='both', direction='in', top=True, labelbottom=False, right=True)

    # Sub Plot
    plt.axes([0.1, 0.1, 0.85, 0.2])  # left, bottom, width, height
    sub_axes = plt.gca()

    sub_axes.errorbar(x=bin_centres, y=data_x - background, yerr=data_x_errors, fmt='ko')
    sub_axes.plot(bin_centres, fit_result.best_fit - background, '-r')
    sub_axes.set_xlim(left=xmin, right=xmax)
    sub_axes.set_xlabel(r'di-photon invariant mass $\mathrm{m_{\gamma\gamma}}$ [GeV]', fontsize=13)
    sub_axes.set_ylabel('Events-Bkg')
    sub_axes.tick_params(which='both', direction='in', top=True, right=True)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the plot
    output_file = os.path.join(output_dir, "invariant_mass_fit.png")
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

    # Show the plot (optional, remove if running in headless mode)
    plt.show()
