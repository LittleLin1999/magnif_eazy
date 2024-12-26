'''
Description: This script is used to run EAZY on the Sapphires photometry catalog.
We run twice using different photometry: CIRC1 and KRON_S.
The script will generate the EAZY input catalog and run EAZY to get the photometric redshifts.


Author: Xiaojing Lin
GitHub: Littlelin1999@gmail.com
Date: 2024-11-11 15:32:27
LastEditors: LittleLin1999 littlelin1999@gmail.com
LastEditTime: 2024-11-26 14:38:04
'''
 

import glob
import os
import numpy as np
 
from astropy.table import Table 
import astropy.units as u


import eazy, astropy
import eazy.hdf5

import time
 
# input and output  
phot_cat_fname = '/data/sapphires/catalogs/4750_v052_merged_phot.fits'
rootname = f'4750_v052'
output_dir = '/home/lxj/data/SAPPHIRE_EAZY/' + rootname


# the code should be run in the output_dir, where templates for the EAZY will be generated
os.makedirs(output_dir, exist_ok=True)
os.chdir(output_dir)

# config
template_Hainline_path = '/home/lxj/magnif_eazy/template_Hainline/'
eazy_photz_path = '/home/lxj/anaconda3/envs/jwst/lib/python3.12/site-packages/eazy/data/'



###----------------- EAZY -----------------####
sexcat = Table.read(phot_cat_fname)

for phot_suffix in ['CIRC1', 'KRON_S']: 

    #----------------- EAZY format -----------------

    eazy_tab = Table()
    eazy_tab['id'] = sexcat['ID']
    eazy_tab['RA'] = sexcat['RA']
    eazy_tab['DEC'] = sexcat['DEC']

    ra_center = np.nanmedian(sexcat['RA'])
    dec_center = np.nanmedian(sexcat['DEC'])
 
    for colname in sexcat.colnames:
        if colname.endswith(f'_{phot_suffix}'): 
            print('Loading', colname)

            err_colname = colname.replace(f'_{phot_suffix}', f'_{phot_suffix}_e')
            en_colname = colname.replace(f'_{phot_suffix}', f'_{phot_suffix}_en')

            ### use the maximum of the two errors
            err = sexcat[err_colname].data
            en = sexcat[en_colname].data
            used_err_colvalue = np.nanmax([err, en], axis=0)

            filt = colname.split('_')[0]
            f = sexcat[colname].data
            e = used_err_colvalue
            

            ### add uncertainty floor: 0.05
            e = np.where( e < 0.05 * f , 0.05 *  f , e)
            nan_flg = np.isnan(f) | np.isnan(en) | (f == 0)  
            e[nan_flg] = np.nan
            f[nan_flg] = np.nan

            eazy_tab['f_{0}'.format(filt)] = np.nan_to_num(f, nan = -999999., posinf=-999999., neginf=-999999.)
            eazy_tab['e_{0}'.format(filt)] = np.nan_to_num(e, nan = -999999., posinf=-999999., neginf=-999999.)

            # if is masked array, fill the masked value with -999999.
            if hasattr(eazy_tab['f_{0}'.format(filt)], 'filled'):
                eazy_tab['f_{0}'.format(filt)] = eazy_tab['f_{0}'.format(filt)].filled(-999999.)
            if hasattr(eazy_tab['e_{0}'.format(filt)], 'filled'):
                eazy_tab['e_{0}'.format(filt)] = eazy_tab['e_{0}'.format(filt)].filled(-999999.)

    eazy_catname = os.path.join(output_dir, f'{rootname}_{phot_suffix}.eazy.cat')
    eazy_tab.write(eazy_catname, format='ascii', overwrite=True)

    
    # ----------------- Run EAZY -----------------
    # generate the template file based on Hailine et al. 2023
    template_seds = glob.glob(os.path.join(template_Hainline_path, '*.sed') )

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
    astropy.io.ascii.write(template_sed_tab, os.path.join(template_Hainline_path, 'template_Hainline.param'), 
                            format="no_header", overwrite=True)
    os.system(f'cp -r {template_Hainline_path} {os.path.join(eazy_photz_path, "templates")}')


    ### reset
    os.system('rm -rf templates')
    os.system('rm FILTER.RES.latest')

    ### soft link of the templates
    eazy.symlink_eazy_inputs(path=eazy_photz_path)



    # PARAMETERS
    params = {}

    ## Filters 
    params['FILTERS_RES'] = os.path.join(eazy_photz_path, 'filters/FILTER.RES.latest')

    ### Templates
    params['TEMPLATES_FILE'] = os.path.join(eazy_photz_path,'templates/template_Hainline/template_Hainline.param')
    #'templates/sfhz/corr_sfhz_13.param'   
    #'templates/template_Hainline/template_Hainline.param'

    ### Error templates
    params['TEMP_ERR_FILE'] = os.path.join(eazy_photz_path,'templates/TEMPLATE_ERROR.v2.0.zfourge')
    params['TEMP_ERR_A2'] = 1.0 # important !!!
    
    #'templates/TEMPLATE_ERROR.eazy_v1.0'
    #'templates/TEMPLATE_ERROR.v2.0.zfourge'
    #'templates/template_error_cosmos2020.txt'
 
    
    mw_ebv = eazy.utils.get_irsa_dust( ra_center, dec_center)  # center of the field
    params['MW_EBV'] = mw_ebv
    params['CAT_HAS_EXTCORR'] = 'n'

    params['ADD_CGM'] = 'n' # turn off the LyA absorption model

    ## Input Files
    params['CATALOG_FILE'] = eazy_catname
    params['CATALOG_FORMAT'] = 'ascii'
    params['OUTPUT_DIRECTORY'] = output_dir
    params['MAIN_OUTPUT_FILE'] = f'{rootname}_{phot_suffix}.eazy'
    params['PRINT_ERRORS'] = 'y'
    params['NOT_OBS_THRESHOLD'] = -90
    params['SYS_ERR'] = 0.0 # already set the uncertainty floor in the input catalog

    ## Redshift / Mag prior
    params['APPLY_PRIOR'] = 'n'
    params['PRIOR_FILE'] = ''
    params['PRIOR_ABZP'] =   (u.nJy).to(u.ABmag) # 23.9 for uJy
    params['PRIOR_FILTER'] = 366

    ## Redshift Grid
    params['Z_MIN'] = 0.01
    params['Z_MAX'] = 30              
    params['Z_STEP'] = 0.01

    
    translate_file = '/home/lxj/magnif_eazy/eazy_translate.txt' # https://eazy-py.readthedocs.io/en/latest/eazy/filters.html

    ### here is Yoshi's setting
    Yoshi_params = {
    "VERBOSITY": 1.0,
    "FILTERS_RES": "FILTER.RES.latest",
    "FILTER_FORMAT": 1.0,
    "SMOOTH_FILTERS": "n",
    "SMOOTH_SIGMA": 100.0,
    "TEMPLATES_FILE": os.path.join(eazy_photz_path,'templates/template_Hainline/template_Hainline.param'),
    "TEMPLATE_COMBOS": "a",
    "NMF_TOLERANCE": 0.0001,
    "WAVELENGTH_FILE": "templates/uvista_nmf/lambda.def",
    "TEMP_ERR_FILE": os.path.join(eazy_photz_path,'templates/TEMPLATE_ERROR.v2.0.zfourge'),
    "TEMP_ERR_A2": 1.0,
    "SYS_ERR": 0.05,
    "APPLY_IGM": "y",
    "IGM_SCALE_TAU": 1.0,
    "SCALE_2175_BUMP": 0.0,
    "TEMPLATE_SMOOTH": 0.0,
    "RESAMPLE_WAVE": "None",
    "MW_EBV": 0.0354,
    "CAT_HAS_EXTCORR": "n",
    "DUMP_TEMPLATE_CACHE": "n",
    "USE_TEMPLATE_CACHE": "n",
    "CACHE_FILE": "photz.tempfilt",
    "CATALOG_FILE": eazy_catname,
    "CATALOG_FORMAT": "ascii",
    "MAGNITUDES": "n",
    "NOT_OBS_THRESHOLD": -90.0,
    "N_MIN_COLORS": 5.0,
    "ARRAY_NBITS": 32.0,
    "OUTPUT_DIRECTORY": output_dir,
    "MAIN_OUTPUT_FILE": f'{rootname}_{phot_suffix}.eazy',
    "PRINT_ERRORS": "y",
    "CHI2_SCALE": 1.0,
    "VERBOSE_LOG": "y",
    "OBS_SED_FILE": "n",
    "TEMP_SED_FILE": "n",
    "POFZ_FILE": "n",
    "BINARY_OUTPUT": "y",
    "APPLY_PRIOR": "n",
    "PRIOR_FILE": "templates/prior_F160W_TAO.dat",
    "PRIOR_FILTER": 366.0,
    "PRIOR_ABZP": 31.4,
    "PRIOR_FLOOR": 0.01,
    "FIX_ZSPEC": "n",
    "Z_MIN": 0.01,
    "Z_MAX": 30.0,
    "Z_STEP": 0.01,
    "Z_STEP_TYPE": 1.0,
    "GET_ZP_OFFSETS": "n",
    "ZP_OFFSET_TOL": 0.0001,
    "REST_FILTERS": "---",
    "RF_PADDING": 1000.0,
    "RF_ERRORS": "n",
    "Z_COLUMN": "z_peak",
    "USE_ZSPEC_FOR_REST": "y",
    "READ_ZBIN": "n",
    "ADD_CGM": "n",
    "H0": 70.0,
    "OMEGA_M": 0.3,
    "OMEGA_L": 0.7}

    # RUN EAZY
    ez = eazy.photoz.PhotoZ(param_file=None, translate_file=translate_file, zeropoint_file=None, 
                            params=params, load_prior=False, load_products=False)
    
    ### save the parameter file
    param_file = os.path.join(output_dir, f'{rootname}_{phot_suffix}.param')
    ez.param.write(param_file)

    # Photometric zeropoint offsets
    NITER = 0 # turn off the iteration
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
                                update_zeropoints=True, iter=iter, n_proc=20, 
                                save_templates=False, error_residuals=False, 
                                NBIN=NBIN, get_spatial_offset=False,
                                )

    # Turn off error corrections derived above
    ez.set_sys_err(positive=True)

    # fit_parallel renamed to fit_catalog 14 May 2021
    ez.fit_catalog(ez.idx, n_proc=20)
    

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
                                                f'Photometry catalog: {phot_cat_fname}',
                                                f'Photometry type: {phot_suffix}',
                                                'Processed by eazy-py',
                                                'Produced by X.Lin and F.Sun for the SAPPHIRES project on %s' % (time.strftime('%Y-%m-%d'))]
                                    
    eazy_photz.write(f'{rootname}_{phot_suffix}_EAZY_photz.ecsv', format='ascii.ecsv', overwrite=True)
