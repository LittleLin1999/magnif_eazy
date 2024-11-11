import h5py


from default_config import *
from scipy.integrate import cumtrapz


import eazy

import eazy.hdf5

from ImgTool.img_utils import magerr2snr, snr2magerr
import os


import matplotlib.patheffects as pe
 
plt.rcParams["font.family"] = "serif"
plt.rcParams['font.size'] = 7.5




def save_eazy_h5(h5_file, clobber=True):
    ez = eazy.hdf5.initialize_from_hdf5(h5_file)

    save_fname = h5_file.replace('.eazy.h5', '.sed_pz.h5')

    
    if os.path.isfile(save_fname): 
        if clobber:
            os.remove(save_fname)
        else:
            print(f'{save_fname} already exists. Skip')
            return
    
    h5 = h5py.File(save_fname, 'w')

    idx_zmap = np.nanargmax(ez.lnp, axis=1)
    z_map = np.array([ez.zgrid[idx] for idx in idx_zmap])
    chi2_map = np.array([chi2_fit[idx] for chi2_fit, idx in zip(ez.chi2_fit, idx_zmap)])


    grp = h5.create_group('BASIC')
    grp.create_dataset('ID', data=ez.cat['id'])
    grp.create_dataset('nusefilt', data=ez.nusefilt)
    grp.create_dataset('z_map', data=z_map)
    grp.create_dataset('chi2_map', data=chi2_map)
    grp.create_dataset('zgrid', data=ez.zgrid)
    grp.create_dataset('lnp', data=ez.lnp)
    grp.create_dataset('chi2', data=ez.chi2_fit)
    

    # do not oversample!
    zlims = ez.pz_percentiles(percentiles=[2.5, 5, 16, 50, 84, 95, 97.5],
                                                oversample=1, selection=None)
    grp.create_dataset('z_2p5', data=zlims[:,0])
    grp.create_dataset('z_5', data=zlims[:,1])
    grp.create_dataset('z_16', data=zlims[:,2])
    grp.create_dataset('z_50', data=zlims[:,3])
    grp.create_dataset('z_84', data=zlims[:,4])
    grp.create_dataset('z_95', data=zlims[:,5])
    grp.create_dataset('z_97p5', data=zlims[:,6])


    grp = h5.create_group('Phot')
    grp.create_dataset('wave_AA', data=ez.pivot)
    grp.create_dataset('fmodel_nJy', data=ez.fmodel)
    grp.create_dataset('efmodel_nJy', data=ez.efmodel)
    grp.create_dataset('fobs_nJy', data=ez.fnu)
    grp.create_dataset('efobs_nJy', data=ez.efnu_orig)


    grp = h5.create_group('Temp') 
    grp.create_dataset('TempName', data=np.array([temp.name for temp in ez.templates], dtype='S100'), )
    grp.create_dataset('TempCoeff', data=ez.fit_coeffs)
    grp.create_dataset('BestFitTempCoeff', data=ez.coeffs_best)
    best_fit_temp = []
    best_fit_wave = []
    for i, idd in enumerate(ez.cat['id']):
        zshow = z_map[i] if z_map[i] >= ez.zgrid[0] else ez.zgrid[0]
        data_ = ez.show_fit(idd, zshow=zshow, get_spec=True, show_fnu=1)
        best_fit_temp.append( (data_['templf'] * data_['flux_unit']).to(u.nJy).value)
        best_fit_wave.append( (data_['templz'] * data_['wave_unit']).to(u.AA).value)
        
    grp.create_dataset('best_fit_temp_nJy', data=best_fit_temp)
    grp.create_dataset('best_fit_wave_AA', data=best_fit_wave)

    h5.close()


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--field', type=str, default='m0416', help='Field name')
    args = parser.parse_args()
    
    field_info_CIRC = {'a370': '/home/lxj/data/JWST_Photometry/MAGNIF/MAGNIF_A370_CIRC1_v2.1.eazy.h5',
                  'm0416': '/home/lxj/data/JWST_Photometry/MAGNIF/MAGNIF_M0416_CIRC1_v2.0.eazy.h5',
                  'a2744': '/home/lxj/data/JWST_Photometry/MAGNIF/MAGNIF_A2744_CIRC1_v1.0.eazy.h5',
                  'm1149': '/home/lxj/data/JWST_Photometry/MAGNIF/MAGNIF_M1149_CIRC1_v1.0.eazy.h5'}
    
    field_info_KRONS = {'a370': '/home/lxj/data/JWST_Photometry/MAGNIF/MAGNIF_A370_KRON_S_v2.1.eazy.h5',
                  'm0416': '/home/lxj/data/JWST_Photometry/MAGNIF/MAGNIF_M0416_KRON_S_v2.0.eazy.h5',
                  'a2744': '/home/lxj/data/JWST_Photometry/MAGNIF/MAGNIF_A2744_KRON_S_v1.0.eazy.h5',
                  'm1149': '/home/lxj/data/JWST_Photometry/MAGNIF/MAGNIF_M1149_KRON_S_v1.0.eazy.h5'}

    h5_file = field_info_CIRC[args.field]
    save_eazy_h5(h5_file, clobber=True)

    h5_file = field_info_KRONS[args.field]
    save_eazy_h5(h5_file, clobber=True)
    
