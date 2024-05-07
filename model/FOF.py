import os
import json
import time
import h5py
import numpy
import scipy
import pyccl
import argparse
from matplotlib import pyplot
from matplotlib.gridspec import GridSpec


# FORMULA
def formula(theta, nu):
    """
    Calculate a complex formula based on the input parameters theta and nu.
    
    Parameters:
        theta (tuple): A tuple containing five elements representing lnf, lng, lnh, lni, and lnj.
        nu (float): A numerical value representing nu.
    
    Returns:
        float: The result of the complex formula calculation.
    """
    lnf, lng, lnh, lni, lnj = theta
    f, g, h, i, j = numpy.exp(lnf), numpy.exp(lng), numpy.exp(lnh), numpy.exp(lni), numpy.exp(lnj)
    
    return 1 - f + f / (1 + numpy.power(g * nu, h)) + i * numpy.power(nu, j)


# MODEL
def model(theta, varphi):
    """
    A function that takes theta and varphi as parameters and calculates a value based on the given formula and scipy integration.
    
    Parameters:
        theta (tuple): A tuple containing five elements representing lnf, lng, lnh, lni, and lnj.
        varphi (tuple): A tuple containing nu and hmf.
        
    Returns:
        array: The calculated result of the model.
    """
    nu, hmf = varphi
    bias = formula(theta, nu)
    
    weight = hmf[:, numpy.newaxis, numpy.newaxis, :] * hmf[numpy.newaxis, :, :, numpy.newaxis]
    factor = bias[:, numpy.newaxis, numpy.newaxis, :] * bias[numpy.newaxis, :, :, numpy.newaxis]
    
    value = scipy.integrate.trapezoid(y=factor * weight, axis=-1)
    value = scipy.integrate.trapezoid(y=value, axis=-1)
    
    product = scipy.integrate.trapezoid(y=weight, axis=-1)
    product = scipy.integrate.trapezoid(y=product, axis=-1)
    
    return value / product


# MAIN
def main(l, n, tag, label):  # noqa: E741
    """
    Function for performing various calculations and generating plots. 
    
    Parameters:
        l (int): The size of the mock box.
        n (int): The number of particles.
        tag (str): The tag of the halo masses.
        mock (str): The type of simulations.
        label (str): The label of the physical model.
        
    Returns:
        result (array): The result of the calculation.
    """
    # MOCK
    name = 'L{}N{}'.format(l, n)
    data_path = '/cosma8/data/dp203/dc-zhan7/HaloCloud/DATA/'
    plot_path = '/cosma8/data/dp203/dc-zhan7/HaloCloud/PLOT/'
    
    # MASS
    m_size = 80
    logm1 = 9.00
    logm2 = 17.0
    delta_m = (logm2 - logm1) / m_size
    m_data = numpy.logspace(logm1 + delta_m / 2, logm2 - delta_m / 2, m_size)
    
    # BIN
    bin_size = 20
    logm1_bin = 11.0
    logm2_bin = 15.0
    delta_bin = (logm2_bin - logm1_bin) / bin_size
    edge_bin = numpy.logspace(logm1_bin, logm2_bin, bin_size + 1)
    center_bin = numpy.logspace(logm1_bin + delta_bin / 2, logm2_bin - delta_bin / 2, bin_size)
    
    # SCALE
    mesh = 3200
    k1 = 2 * numpy.pi / (l / 2)
    k2 = 2 * numpy.pi / l * (mesh / 2)
    
    k_size = 100
    logk1 = numpy.log10(k1)
    logk2 = numpy.log10(k2)
    delta_k = (logk2 - logk1) / k_size
    k_data = numpy.logspace(logk1 + delta_k / 2, logk2 - delta_k / 2, k_size)
    
    # COSMO
    os.makedirs(plot_path + name + '_' + label + '/PHH/' + tag, exist_ok=True)
    with open(data_path + name + '_' + label + '/INFO/COSMO.json', 'r') as file:
        cosmo = json.load(file)
    
    cosmo_ccl = pyccl.cosmology.Cosmology(
        h=cosmo['H'],
        w0=cosmo['W0'],
        wa=cosmo['WA'],
        n_s=cosmo['NS'],
        m_nu=cosmo['MNU'],
        Neff=cosmo['NEFF'],
        T_CMB=cosmo['TCMB'],
        mass_split='single',
        sigma8=cosmo['SIGMA8'],
        Omega_k=cosmo['OMEGAK'],
        Omega_c=cosmo['OMEGAC'],
        Omega_b=cosmo['OMEGAB'],
        mg_parametrization=None,
        matter_power_spectrum='halofit',
        transfer_function='boltzmann_camb',
        extra_parameters={
            'camb': {'kmax': 10000, 'lmax': 10000, 'halofit_version': 'mead2020_feedback', 'HMCode_logT_AGN': 7.8}}
    )
    
    # DATA
    result = {}
    length = 0
    mock_size = 78
    with h5py.File(data_path + name + '_' + label + '/PHH/DMO_' + tag + '/ALPHA_MATH.hdf5', 'r') as file:
        chain = file['CHAIN'][:].astype('float32')
        value = file['VALUE'][:].astype('float32')
        theta = chain[numpy.argmax(value), :]
        print(theta)
    
    with h5py.File(data_path + 'L1000N1800_' + label + '/PHH/DMO_' + tag + '/ALPHA_MATH.hdf5', 'r') as file:
        chain0 = file['CHAIN'][:].astype('float32')
        value0 = file['VALUE'][:].astype('float32')
        theta0 = chain0[numpy.argmax(value0), :]
        print(theta0)
    
    with h5py.File(data_path + 'L1000N1800_' + label + '/HMF/DMO_' + tag + '/MATH.hdf5', 'r') as file:
        hmf_chain0 = file['CHAIN'][:].astype('float32')
        hmf_value0 = file['VALUE'][:].astype('float32')
        hmf_theta0 = hmf_chain0[numpy.argmax(hmf_value0), :]
        print(hmf_theta0)
    
    for k in range(mock_size, mock_size - length - 1, -1):
        start = time.time()
        zid = str(k).zfill(4)
        print('ID: {}'.format(zid))
        os.makedirs(plot_path + name + '_' + label + '/PHH/' + tag + '/' + zid, exist_ok=True)
        
        with open(data_path + name + '_' + label + '/INFO/HALO.json', 'r') as file:
            z = json.load(file)[zid]
        
        with h5py.File(data_path + name + '_' + label + '/PHH/DMO_' + tag + '/POWER.hdf5', 'r') as file:
            noise_dmo = file[zid]['NOISE'][:].astype('float32')
        
        with h5py.File(data_path + name + '_' + label + '/HMF/DMO_' + tag + '/DATA.hdf5', 'r') as file:
            hmf_dmo = file[zid]['MEAN'][:].astype('float32')
            
        with h5py.File(data_path + name + '_' + label + '/PHH/DMO_' + tag + '/DATA.hdf5', 'r') as file:
            mean_dmo = file[zid]['MEAN'][:].astype('float32')
            scatter_dmo = file[zid]['SCATTER'][:].astype('float32')
        
        with h5py.File(data_path + name + '_' + label + '/PHH/HYDRO_' + tag + '/DATA.hdf5', 'r') as file:
            mean_hydro = file[zid]['MEAN'][:].astype('float32')
            scatter_hydro = file[zid]['SCATTER'][:].astype('float32')
        
        # ALPHA
        m_select = noise_dmo > 0
        k_select = (0.02 <= k_data) & (k_data <= 0.08)
        pk_data = pyccl.power.linear_matter_power(cosmo=cosmo_ccl, k=k_data, a=1 / (1 + z))
        
        sample_size = 10000
        m_select_size = len(m_select[m_select])
        k_select_size = len(k_select[k_select])     
        bin_index1, bin_index2 = numpy.meshgrid(numpy.arange(m_select_size), numpy.arange(m_select_size), indexing='ij')
        
        bin_index1 = bin_index1[:, :, numpy.newaxis, numpy.newaxis]
        bin_index2 = bin_index2[:, :, numpy.newaxis, numpy.newaxis]
        sample_index = numpy.random.choice(k_select_size, (k_select_size, sample_size), replace=True)
        
        # ALPHA DMO
        data_dmo = mean_dmo[m_select, :, :][:, m_select, :] / pk_data
        error_dmo = scatter_dmo[m_select, :, :][:, m_select, :] / pk_data
        
        sample_error_dmo = error_dmo[:, :, k_select][bin_index1, bin_index2, sample_index]
        sample_data_dmo = data_dmo[:, :, k_select][bin_index1, bin_index2, sample_index]
        
        sample_dmo = numpy.random.normal(loc=sample_data_dmo, scale=sample_error_dmo)
        alpha_dmo = numpy.median(numpy.median(sample_dmo, axis=2), axis=2)
        sigma_dmo = numpy.median(numpy.std(sample_dmo, axis=2), axis=2)
        
        bias_data_dmo = numpy.sqrt(numpy.diagonal(alpha_dmo))
        bias_error_dmo = numpy.diagonal(sigma_dmo) / (2 * bias_data_dmo)
        
        # ALPHA HYDRO
        data_hydro = mean_hydro[m_select, :, :][:, m_select, :] / pk_data
        error_hydro = scatter_hydro[m_select, :, :][:, m_select, :] / pk_data
        
        sample_error_hydro = error_hydro[:, :, k_select][bin_index1, bin_index2, sample_index]
        sample_data_hydro = data_hydro[:, :, k_select][bin_index1, bin_index2, sample_index]
        
        sample_hydro = numpy.random.normal(loc=sample_data_hydro, scale=sample_error_hydro)
        alpha_hydro = numpy.median(numpy.median(sample_hydro, axis=2), axis=2)
        sigma_hydro = numpy.median(numpy.std(sample_hydro, axis=2), axis=2)
        
        bias_data_hydro = numpy.sqrt(numpy.diagonal(alpha_hydro))
        bias_error_hydro = numpy.diagonal(sigma_hydro) / (2 * bias_data_hydro)
        
        # BIAS
        mass_range = (10 ** logm1_bin < m_data) & (m_data < edge_bin[1:][m_select].max())
        hmf_data = numpy.reshape(hmf_dmo[mass_range], (m_select_size, len(hmf_dmo[mass_range]) // m_select_size))
        
        delta_c = 1.686
        nu_data = delta_c / cosmo_ccl.sigmaM(M=m_data[mass_range], a=1 / (1 + z))
        nu_data = numpy.reshape(nu_data, (m_select_size, len(nu_data) // m_select_size))
        
        eta_f, eta_g, eta_h, eta_i, eta_j, zeta_f, zeta_g, zeta_h, zeta_i, zeta_j = theta
        lnf = eta_f + zeta_f * numpy.log(1 + z)
        lng = eta_g + zeta_g * numpy.log(1 + z)
        lnh = eta_h + zeta_h * numpy.log(1 + z)
        lni = eta_i + zeta_i * numpy.log(1 + z)
        lnj = eta_j + zeta_j * numpy.log(1 + z)
        
        eta_f0, eta_g0, eta_h0, eta_i0, eta_j0, zeta_f0, zeta_g0, zeta_h0, zeta_i0, zeta_j0 = theta0
        lnf0 = eta_f0 + zeta_f0 * numpy.log(1 + z)
        lng0 = eta_g0 + zeta_g0 * numpy.log(1 + z)
        lnh0 = eta_h0 + zeta_h0 * numpy.log(1 + z)
        lni0 = eta_i0 + zeta_i0 * numpy.log(1 + z)
        lnj0 = eta_j0 + zeta_j0 * numpy.log(1 + z)
        
        eta_a0, eta_b0, eta_c0, eta_d0, eta_e0, zeta_a0, zeta_b0, zeta_c0, zeta_d0, zeta_e0 = hmf_theta0
        lna0 = eta_a0 + zeta_a0 * numpy.log(1 + z)
        lnb0 = eta_b0 + zeta_b0 * numpy.log(1 + z)
        lnc0 = eta_c0 + zeta_c0 * numpy.log(1 + z)
        lnd0 = eta_d0 + zeta_d0 * numpy.log(1 + z)
        lne0 = eta_e0 + zeta_e0 * numpy.log(1 + z)
        
        lna0 = numpy.log(0.88 * delta_c)
        vartheta = (lnf, lng, lnh, lni, lnj)
        vartheta0 = (lnf0, lng0, lnh0, lni0, lnj0)
        hmf_vartheta0 = (lnc0 - lna0, lnb0, lnc0, lnd0 + lne0 - lna0, lne0)
        
        bias_formula = formula(vartheta, delta_c / cosmo_ccl.sigmaM(M=m_data, a=1 / (1 + z)))
        bias_formula0 = formula(vartheta0, delta_c / cosmo_ccl.sigmaM(M=m_data, a=1 / (1 + z)))
        hmf_bias_formula0 = formula(hmf_vartheta0, delta_c / cosmo_ccl.sigmaM(M=m_data, a=1 / (1 + z)))
        
        bias_model = numpy.sqrt(numpy.diagonal(model(theta=vartheta, varphi=(nu_data, hmf_data))))
        bias_model0 = numpy.sqrt(numpy.diagonal(model(theta=vartheta0, varphi=(nu_data, hmf_data))))
        
        bias1 = pyccl.halos.hbias.sheth99.HaloBiasSheth99(mass_def='fof', mass_def_strict=True)(cosmo=cosmo_ccl, M=m_data, a=1 / (1 + z))
        bias2 = pyccl.halos.hbias.sheth01.HaloBiasSheth01(mass_def='fof', mass_def_strict=True)(cosmo=cosmo_ccl, M=m_data, a=1 / (1 + z))
        bias3 = pyccl.halos.hbias.tinker10.HaloBiasTinker10(mass_def='vir', mass_def_strict=True)(cosmo=cosmo_ccl, M=m_data, a=1 / (1 + z))
        bias4 = pyccl.halos.hbias.bhattacharya11.HaloBiasBhattacharya11(mass_def='fof', mass_def_strict=True)(cosmo=cosmo_ccl, M=m_data, a=1 / (1 + z))                                                                                  
        
        ratio1 = numpy.divide(bias1, bias_formula0, out=numpy.zeros_like(bias1), where=bias_formula0 != 0)
        ratio2 = numpy.divide(bias2, bias_formula0, out=numpy.zeros_like(bias2), where=bias_formula0 != 0)
        ratio3 = numpy.divide(bias3, bias_formula0, out=numpy.zeros_like(bias3), where=bias_formula0 != 0)
        ratio4 = numpy.divide(bias4, bias_formula0, out=numpy.zeros_like(bias4), where=bias_formula0 != 0)
        ratio = numpy.divide(bias_formula, bias_formula0, out=numpy.zeros_like(bias_formula), where=bias_formula0 != 0)
        hmf_ratio0 = numpy.divide(hmf_bias_formula0, bias_formula0, out=numpy.zeros_like(hmf_bias_formula0), where=bias_formula0 != 0)
        
        ratio_dmo = numpy.divide(bias_data_dmo, bias_model0, out=numpy.zeros_like(bias_data_dmo), where=bias_model0 != 0)
        varsigma_dmo = numpy.divide(bias_error_dmo, bias_model0, out=numpy.zeros_like(bias_error_dmo), where=bias_model0 != 0)
        
        ratio_hydro = numpy.divide(bias_data_hydro, bias_model0, out=numpy.zeros_like(bias_data_hydro), where=bias_model0 != 0)
        varsigma_hydro = numpy.divide(bias_error_hydro, bias_model0, out=numpy.zeros_like(bias_error_hydro), where=bias_model0 != 0)
        
        # PLOT
        pyplot.rcParams['font.size'] = 20
        pyplot.rcParams['text.usetex'] = True
        figure = pyplot.figure(figsize=(12, 12))
        gridspec = GridSpec(12, 12, figure=figure, wspace=0.0, hspace=0.0)
        
        plot = figure.add_subplot(gridspec[:8, :])
        
        plot.plot(m_data, bias_formula0, color='black', linestyle='-', linewidth=2.0, label=r'$\mathrm{L1000N1800}$')
        
        plot.plot(m_data, bias_formula, color='black', linestyle='--', linewidth=2.0, label=r'$\mathrm{L1000N3600}$')
        
        plot.plot(m_data, hmf_bias_formula0, color='black', linestyle=':', linewidth=2.0, label=r'$\mathrm{L1000N1800} \: \mathrm{HMF}$')
        
        plot.plot(m_data, bias1, color='purple', linestyle='-', linewidth=2.0, label=r'$\mathrm{Sheth+99}$')
        
        plot.plot(m_data, bias2, color='green', linestyle='-', linewidth=2.0, label=r'$\mathrm{Sheth+01}$')
        
        plot.plot(m_data, bias3, color='orange', linestyle='-', linewidth=2.0, label=r'$\mathrm{Tinker+10}$')
        
        plot.plot(m_data, bias4, color='brown', linestyle='-', linewidth=2.0, label=r'$\mathrm{Bhattacharya+11}$')
        
        plot.errorbar(x=center_bin[m_select], y=bias_data_hydro, yerr=bias_error_hydro, fmt='s', ecolor='pink', capthick=2.5, capsize=2.5, markersize=15, markeredgewidth=2.5, markerfacecolor='none', markeredgecolor='darkred', label=r'$\mathrm{HYDRO}$')
        
        plot.errorbar(x=center_bin[m_select], y=bias_data_dmo, yerr=bias_error_dmo, fmt='s', ecolor='skyblue', capthick=2.5, capsize=2.5, markersize=15, markeredgewidth=2.5, markerfacecolor='none', markeredgecolor='darkblue', label=r'$\mathrm{DMO}$')
        
        plot.set_xscale('log')
        plot.set_yscale('log')
        
        plot.legend(fontsize=20)
        plot.set_title(r'$z = {:.2f}$'.format(z))
        
        plot.set_xlim(10 ** 10.0, 10 ** 16.0)
        plot.set_ylim(bias_model.min() / 2, bias_model.max() * 2)
        
        plot.set_xticklabels([])
        plot.get_yticklabels()[0].set_visible([])
        plot.set_ylabel(r'$b_\mathrm{h}^\mathrm{L} (M)$')
        
        plot = figure.add_subplot(gridspec[8:, :])
        plot.plot(m_data, numpy.ones(m_size), color='black', linestyle='-', linewidth=2.0)
        
        plot.plot(m_data, numpy.ones(m_size) * (1.0 - 0.1), color='grey', linestyle='--', linewidth=2.0)
        
        plot.plot(m_data, numpy.ones(m_size) * (1.0 + 0.1), color='grey', linestyle='--', linewidth=2.0)
        
        plot.plot(m_data, ratio, color='black', linestyle='--', linewidth=2.0)
        
        plot.plot(m_data, hmf_ratio0, color='black', linestyle=':', linewidth=2.0)
        
        plot.plot(m_data, ratio1, color='purple', linestyle='-', linewidth=2.0)
        
        plot.plot(m_data, ratio2, color='green', linestyle='-', linewidth=2.0)
        
        plot.plot(m_data, ratio3, color='orange', linestyle='-', linewidth=2.0)
        
        plot.plot(m_data, ratio4, color='brown', linestyle='-', linewidth=2.0)
        
        plot.errorbar(x=center_bin[m_select], y=ratio_hydro, yerr=varsigma_hydro, fmt='s', ecolor='pink', capthick=2.5, capsize=2.5, markersize=15, markeredgewidth=2.5, markerfacecolor='none', markeredgecolor='darkred')
        
        plot.errorbar(x=center_bin[m_select], y=ratio_dmo, yerr=varsigma_dmo, fmt='s', ecolor='skyblue', capthick=2.5, capsize=2.5, markersize=15, markeredgewidth=2.5, markerfacecolor='none', markeredgecolor='darkblue')
        
        plot.set_ylim(1.0 - 0.18, 1.0 + 0.18)
        plot.set_xlim(10 ** 10.0, 10 ** 16.0)
        
        plot.set_xscale('log')
        plot.set_xlabel(r'$M_1^\mathrm{' + tag.lower() + r'} \: [M_\odot]$')
        
        figure.savefig(plot_path + name + '_' + label + '/PHH/' + tag + '/' + zid + '/ALPHA.pdf', bbox_inches='tight')
        pyplot.close(figure)
        
        result[zid] = numpy.exp(vartheta)
        print('TIME: {:.2f} minutes'.format((time.time() - start) / 60))
    return result


if __name__ == '__main__':
    # INPUT
    PARSE = argparse.ArgumentParser(description='Halo Mass Function')
    PARSE.add_argument('--l', type=int, default=None, help='Mock Box Size')
    PARSE.add_argument('--n', type=int, default=None, help='Number of Particles')
    PARSE.add_argument('--tag', type=str, default=None, help='Tag of Halo Masses')
    PARSE.add_argument('--mock', type=str, default=None, help='Type of Simulations')
    PARSE.add_argument('--label', type=str, default=None, help='Label of Physical Model')
    
    L = PARSE.parse_args().l
    N = PARSE.parse_args().n
    TAG = PARSE.parse_args().tag
    MOCK = PARSE.parse_args().mock
    LABEL = PARSE.parse_args().label
    
    # FLAMINGO
    print(L, N, TAG, LABEL)
    RESULT = main(L, N, TAG, LABEL)
