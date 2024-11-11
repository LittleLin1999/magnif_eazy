'''
Description: This script is used to run EAZY on the Sapphires photometry catalog.
We run twice using different photometry: CIRC1 and KRON_S.
The script will generate the EAZY input catalog and run EAZY to get the photometric redshifts.


Author: Xiaojing Lin
GitHub: Littlelin1999@gmail.com
Date: 2024-11-11 15:32:27
LastEditors: LittleLin1999 littlelin1999@gmail.com
LastEditTime: 2024-11-11 15:37:45
'''
 

import glob
import os
from astropy.io import fits
import numpy as np
import multiprocessing  
from astropy.table import Table, vstack
import astropy.units as u
from matplotlib.patches import Circle, Ellipse, Rectangle, Polygon
from photutils.aperture import EllipticalAperture, CircularAperture, aperture_photometry

import tqdm

import eazy, astropy
import eazy.hdf5

import time

from scipy.integrate import cumtrapz

 
phot_cat_fname = '/data/sapphires/catalogs/4750_v03_merged_phot.fits'
rootname = f'4750_v03'

output_dir = '/home/lxj/data/SAPPHIRE_EAZY/' + rootname

###----------------- EAZY -----------------####
sexcat = Table.read(phot_cat_fname)

for suffix in ['CIRC1', 'KRON_S']: 

    #----------------- EAZY format -----------------

    eazy_tab = Table()
    eazy_tab['ID'] = sexcat['ID']
    eazy_tab['RA'] = sexcat['RA']
    eazy_tab['DEC'] = sexcat['DEC']
 
    for colname in sexcat.colnames:
        if colname.endswith(f'_{suffix}'): 
            print('Loading', colname)

            err_colname = colname.replace(f'_{suffix}', f'_{suffix}_e')
            en_colname = colname.replace(f'_{suffix}', f'_{suffix}_en')

            ### use the maximum of the two errors
            used_err_colvalue = np.nanmax([sexcat[err_colname].data, sexcat[en_colname].data], axis=0)
            
            filt = colname.split('_')[0]
            f = sexcat[colname].data
            e = used_err_colvalue
            

            ### add uncertainty floor: 0.05
            e = np.where( e < 0.05 * f , 0.05 *  f , e)
            e[f == 0] = np.nan
            f[f == 0] = np.nan


            eazy_tab['f_{0}'.format(filt)] = np.nan_to_num(f, nan = -999999.)
            eazy_tab['e_{0}'.format(filt)] = np.nan_to_num(e, nan = -999999.)

    eazy_catname = os.path.join(output_dir, f'{rootname}.eazy.cat')
    eazy_tab.write(eazy_catname, format='ascii', overwrite=True)


    # ----------------- Run EAZY -----------------
    if  os.path.exists('templates'): os.system('rm -rf templates')
    if (not os.path.exists('templates/template_Hainline')):
            # generate the template file based on Hailine et al. 2023
            template_seds = glob.glob('/home/lxj/data/JWST_Photometry/MAGNIF/template_Hainline/*.sed')

            # sorted by row number
            nrow_seds = [len(np.genfromtxt(s)) for s in template_seds]
            order = np.argsort(nrow_seds)[::-1]
            template_seds = np.array(template_seds)[order]
            template_seds = list(template_seds[-8:]) + list(template_seds[:-8])
            ### let the 2818 grid template be the first one
            ### if the first too sparse / too fine ==> leading to interpolation abnormals

            template_sed_tab = Table()
            template_sed_tab['id'] = np.arange(1, len(template_seds)+1)
            template_sed_tab['file'] = np.array(['templates/template_Hainline/' + os.path.basename(s) for s in template_seds])
            template_sed_tab['Lambda_conv'] = 1.0
            template_age = np.zeros(len(template_seds))
            # template_age[9] = 0.025
            # template_age[10] = 0.005
            template_sed_tab['Age'] = template_age
            template_sed_tab['Template_err'] = 1.0
            astropy.io.ascii.write(template_sed_tab,'/home/lxj/data/JWST_Photometry/MAGNIF/template_Hainline/template_Hainline.param', format="no_header", overwrite=True)

            #os.system('cp template_Hainline.param /home/lxj/data/JWST_Photometry/MAGNIF/template_Hainline/')

            os.system('cp -r template_Hainline /home/lxj/jwst/eazy-photoz/templates/')
            
            # eazy.symlink_eazy_inputs()
            # os.system('rm -rf templates')
            # os.symlink('/home/lxj/jwst/eazy-photoz/templates/', 'templates')
    os.system('rm -rf templates')
    os.system('rm FILTER.RES.latest')
    eazy.symlink_eazy_inputs(path='/home/lxj/jwst/eazy-photoz/')        



    # PARAMETERS
    params = {}

    ## Filters 
    params['FILTERS_RES'] = '/home/lxj/jwst/eazy-photoz/filters/FILTER.RES.latest'

    ### Templates
    params['TEMPLATES_FILE'] = '/home/lxj/jwst/eazy-photoz/templates/template_Hainline/template_Hainline.param'
    #'/home/lxj/jwst/eazy-photoz/templates/sfhz/corr_sfhz_13.param'   
    #'/home/lxj/jwst/eazy-photoz/templates/template_Hainline/template_Hainline.param'

    params['TEMP_ERR_FILE'] = '/home/lxj/jwst/eazy-photoz/templates/TEMPLATE_ERROR.v2.0.zfourge'
    # '/home/lxj/jwst/eazy-photoz/templates/TEMPLATE_ERROR.eazy_v1.0'
    #'/home/lxj/jwst/eazy-photoz/templates/TEMPLATE_ERROR.v2.0.zfourge'
    #'/home/lxj/jwst/eazy-photoz/templates/template_error_cosmos2020.txt'
    
    params['MW_EBV'] = eazy.utils.get_irsa_dust( field_center[field][0], field_center[field][1]) # center of the field
    params['CAT_HAS_EXTCORR'] = 'n'

    ## Input Files
    params['CATALOG_FILE'] = eazy_catname
    params['CATALOG_FORMAT'] = 'ascii'
    params['OUTPUT_DIRECTORY'] = '/home/lxj/data/JWST_Photometry/MAGNIF/catalog/'
    params['MAIN_OUTPUT_FILE'] = f'{rootname}.eazy'
    params['PRINT_ERRORS'] = 'y'
    params['NOT_OBS_THRESHOLD'] = -90

    ## Redshift / Mag prior
    params['APPLY_PRIOR'] = 'n'
    params['PRIOR_FILE'] = ''
    params['PRIOR_ABZP'] =   (u.nJy).to(u.ABmag) # 23.9 for uJy
    params['PRIOR_FILTER'] = 366

    ## Redshift Grid
    params['Z_MIN'] = 0.01
    params['Z_MAX'] = 30              
    params['Z_STEP'] = 0.01



    translate_file = '/home/lxj/data/JWST_Photometry/eazy_translate.txt' # https://eazy-py.readthedocs.io/en/latest/eazy/filters.html


    # RUN EAZY
    ez = eazy.photoz.PhotoZ(param_file=None, translate_file=translate_file, zeropoint_file=None, 
                            params=params, load_prior=False, load_products=False)

    # Photometric zeropoint offsets
    NITER = 5
    NBIN = np.minimum(ez.NOBJ//100, 180)
    ez.param.params['VERBOSITY'] = 1.

    for iter in range(NITER):
        print('Photometric zeropoint offsets, iteration: %d' %(iter+1))
        
        # select high SNR objects
        sn = ez.fnu/ez.efnu
        F200W_sn = sn[:, list(ez.f_numbers).index(366)]
        clip = (sn > 5).sum(axis=1) > 5
        clip &= F200W_sn > 5
        
        ez.iterate_zp_templates(idx=ez.idx[clip], update_templates=False, 
                                update_zeropoints=True, iter=iter, n_proc=128, 
                                save_templates=False, error_residuals=False, 
                                NBIN=NBIN, get_spatial_offset=False,
                                )

    # Turn off error corrections derived above
    ez.set_sys_err(positive=False)

    # fit_parallel renamed to fit_catalog 14 May 2021
    ez.fit_catalog(ez.idx, n_proc=128)
    

    # Write out the results
    eazy.hdf5.write_hdf5(ez, h5file=ez.param['MAIN_OUTPUT_FILE'] + '.h5', include_fit_coeffs=True)

    ### save ez object
    # Xiaojing's Note: do not oversample! 
    zlimits = ez.pz_percentiles(percentiles=[2.5, 5,16,50,84,95, 97.5],
                                        oversample=1) 
    z_2p5_array = zlimits[:,0]
    z_5_array = zlimits[:,1]
    z_16_array = zlimits[:,2]
    z_50_array = zlimits[:,3]
    z_84_array = zlimits[:,4]
    z_95_array = zlimits[:,5]
    z_97p5_array = zlimits[:,6]
    
    # XLin's Note:
    # `get_maxlnp_redshift` in eazy-py/photoz.py: 
    # z_best is set to be -1 if the maximum P(z) (or minimum chi2 equally, if eazy-py calculate Pz correctly) is the first one of the given z grid.  
    # Then, in the `get_maxlnp_redshift` an analytic parabola fit is performed around the z_best, yielding a new z_best (and lnpmax) from the parabola formula. 
    
    # Do not use ez.zbest. Instead, calculate the maximum P(z) redshift by ourselves.
    # Note that ez.lnp = p(z) from chi2 * template_error_function

    idx_zmap = np.nanargmax(ez.lnp, axis=1)
    z_map = np.array([ez.zgrid[idx] for idx in idx_zmap])
    chi2_map = np.array([chi2_fit[idx] for chi2_fit, idx in zip(ez.chi2_fit, idx_zmap)])

    eazy_photz = Table()
    eazy_photz['ID'] = ez.cat['id']
    eazy_photz['z_map'] = z_map
    eazy_photz['chi2_map'] = chi2_map
    eazy_photz['z_50'] = z_50_array
    eazy_photz['z_16'] = z_16_array
    eazy_photz['z_84'] = z_84_array
    eazy_photz['z_5'] = z_5_array
    eazy_photz['z_95'] = z_95_array
    eazy_photz['z_2p5'] = z_2p5_array
    eazy_photz['z_97p5'] = z_97p5_array
    eazy_photz['nusefilt'] = ez.nusefilt

    eazy_photz.meta = {}
    eazy_photz.meta['columns'] = {'ID': 'ID of the object',
                                    'z_map': 'The redshift with the maximum P(z)',
                                    'chi2_map': 'The chi2 value at the redshift with the maximum P(z)',
                                    'z_50': '50th percentile of the redshift probability distribution',
                                    'z_16': '16th percentile of the redshift probability distribution',
                                    'z_84': '84th percentile of the redshift probability distribution',
                                    'z_5': '5th percentile of the redshift probability distribution',
                                    'z_95': '95th percentile of the redshift probability distribution',
                                    'z_2p5': '2.5th percentile of the redshift probability distribution',
                                    'z_97p5': '97.5th percentile of the redshift probability distribution',}
    eazy_photz.meta['comment'] =  ['EAZY template from Hainline et al. 2023',
                                                f'Photometry catalog: MAGNIF {field} {tmp_version}',
                                                'Photometry type: CIRC1',
                                                'Processed by eazy-py',
                                                'Produced by X.Lin and F.Sun for the MAGNIF project on %s' % (time.strftime('%Y-%m-%d'))]
                                    
    eazy_photz.write(f'MAGNIF_{field}_{suffix}_{tmp_version}_EAZY_photz.ecsv', format='ascii.ecsv', overwrite=True)
