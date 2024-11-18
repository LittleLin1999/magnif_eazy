

import numpy as np
import glob
import os
import time

from collections import OrderedDict

from scipy.integrate import cumtrapz

import eazy
import eazy.hdf5

from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy import wcs
from astropy.table import Table, vstack
from astropy.io import fits
from astropy.visualization import (MinMaxInterval, SqrtStretch, ImageNormalize)
from astropy.visualization.mpl_normalize import simple_norm
from astropy.visualization import ZScaleInterval
from astropy.nddata import Cutout2D, NDData
from astropy.wcs import WCS

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["font.family"] = "serif"
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['xtick.direction'] = 'in'  
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams["xtick.minor.visible"] =  True
plt.rcParams["ytick.minor.visible"] =  True
plt.rcParams["xtick.top"] =  True
plt.rcParams["ytick.right"] =  True
plt.rcParams["ytick.left"] =  True
plt.rcParams['font.size'] = 8

import warnings
warnings.filterwarnings('ignore')

def magerr2snr(magerr):
    nsr = 10**(magerr / 2.5) - 1
    return 1 / nsr

def snr2magerr(snr):
    magerr = 2.5 * np.log10(1 + 1/snr)
    return magerr

 


def show_eazy_result(id, save_dir=None, save_suffix='EAZY', 
                     fieldname='A2744', phot_ver='v1p0', phot_type='CIRC1'):

    global image_dict, phot, ez
    global z_5_array, z_16_array, z_50_array, z_84_array, z_95_array, z_map, chi2_map

    ez_idx = np.where(ez.cat['id'] == id)[0][0]

    # figure design
    nrow = len(image_dict) // 10 
    nrow += 1 if len(image_dict) % 10 else 0

    fig =  plt.figure(figsize=(6, 2 + nrow * 0.5))
    gs = fig.add_gridspec(2 + nrow, 10, height_ratios=[3,0.5] + [1.25] * nrow, wspace=0.02, hspace=0.2)

     
    # load data
    if z_map[ez_idx] <= ez.zgrid[0]: zshow = ez.zgrid[0] # failed fit, keep the starting point
    else: zshow = None
    
    data = ez.show_fit(id, zshow=zshow, 
                            show_components=False, axes = None,   show_missing=False,
                                logpz=True, zr=[0,40], show_fnu=1 ,with_tef=True,
                            get_spec=True    )

  
    wv_obs = ((data['pivot']) * data['wave_unit']).to(u.um).value

    fobs = np.where(data['fobs'] <= 0, np.nan, data['fobs'])
    efobs = np.where(data['fobs'] <= 0, np.nan, data['efobs'])
    snr_obs = fobs / efobs

    fobs = (fobs * data['flux_unit']).to(u.ABmag).value
    efobs = snr2magerr(snr_obs)

    fmodel = (data['model'] * data['flux_unit']).to(u.ABmag).value
    snr_model = data['model'] /  data['emodel'] 
    efmodel = snr2magerr(snr_model)
    efmodel[efmodel < 0] = np.nan # set to nan if magerr < 0 ==> invalid points

    
    # show the best fit SED
    ax = fig.add_subplot(gs[0, 0:4])
    ax.errorbar(wv_obs, fobs, yerr=efobs, lw=1,
                color='r', marker='o', ms=2.5,linestyle='None',zorder=5, capsize=1., label='Observed')
    try:
        ax.errorbar(wv_obs, fmodel, yerr=efmodel, lw=0.85,
                marker='s', mfc='none', color='b', ls='none', ms=3,zorder=4, label='Model')
    except:
        print('MODEL ERROR WARNING on source %d'%id, ': efmodel = ', efmodel)
        efmodel = efmodel[efmodel > 0]
        ax.errorbar(wv_obs, fmodel, yerr=efmodel, 
                marker='s', mfc='none', color='b', ls='none', ms=3,zorder=4, label='Model')
    wv_template = ((data['templz']) * data['wave_unit']).to(u.um).value
    f_template = ((data['templf']) * data['flux_unit']).to(u.ABmag).value
    f_template[f_template>40] = 99
    xlim = wv_obs.min() - 0.5, wv_obs.max() + 0.5
    flg  = ((wv_template > xlim[0]) & (wv_template < xlim[1]))
    ax.plot(wv_template[flg], f_template[flg], color='lightblue',   lw=1.)
    ylim = np.nanmin(fobs) -1, np.nanmax(fobs) + 1
    ax.set_ylim(ylim)
    ax.margins(x=0, y=0)
    ax.invert_yaxis()
    _ = ax.set_ylabel('AB Magnitude')
    _ = ax.set_xlabel(r'Wavelength ($\mu$m)',  labelpad=-1.)

    _ = ax.text(0.1, 0.9,
            r'$z_{\rm MAP}$=%.2f' % (z_map[ez_idx]),
            fontsize=7, transform=ax.transAxes, color='k',  
            zorder=100,
            ha='left', va='top', bbox=dict(facecolor='w', alpha=0.5, edgecolor='none'))

    ra, dec = ez.cat[ez_idx]['ra'], ez.cat[ez_idx]['dec']
    ra = '%s%.5f'%(('+' if ra > 0 else ''), ra)
    dec = '%s%.5f'%(('+' if dec > 0 else ''), dec)
    ax.set_title(f'{fieldname}{ra}{dec}   ID  {id}',  fontsize=7 )


    # show the chi2
    pz = np.exp(ez.lnp[ez_idx])
    z_16 = z_16_array[ez_idx]
    z_84 = z_84_array[ez_idx]
    z_5 = z_5_array[ez_idx]
    z_95 = z_95_array[ez_idx]


    ax = fig.add_subplot(gs[0, 5:7],  )
    flg = (ez.zgrid >= 0) & (ez.zgrid <= np.nanmax([20,z_map[ez_idx]+2 ]))
    ax.plot(ez.zgrid[flg], ez.chi2_fit[ez_idx][flg], color='orange', lw=1.)
    # ax.set_yscale('log')
    ax.axvline(z_map[ez_idx], color='g', linestyle='--', lw=0.75)
    ax.set_ylabel(r'$\chi^2$', fontsize=10, labelpad=-1.)
    ax.set_xlabel('$z$', fontsize=10., labelpad=-2.)
    ax.text(0.1, 0.9,
            r'$\chi^2_{\rm MAP}$=%.3f' % (chi2_map[ez_idx]),
            fontsize=7, transform=plt.gca().transAxes, color='k',  
            ha='left', va='top', bbox=dict(facecolor='w', alpha=0.5, edgecolor='none'))
    ax.set_title(f'{phot_ver} {phot_type}', fontsize=7 )   

    ax.axvspan(z_5, z_95, color='g', alpha=0.05, zorder=-1)
    ax.axvspan(z_16, z_84, color='g', alpha=0.25, zorder=-1)


    # inner plot to show the p(z)
    # ins = plt.gca().inset_axes([0.7,0.15,0.25,0.25])
    ins = fig.add_subplot(gs[0, 8:])
    ins.plot(ez.zgrid, pz, color='orange', zorder=1, lw=1.)
    zlim1 = (ez.zgrid > z_map[ez_idx] - 3 ) & (ez.zgrid < z_map[ez_idx] + 3)
    zlim2 = (pz > 1e-30)
    zlim = zlim1 | zlim2
    zlim = (ez.zgrid[zlim].min(), ez.zgrid[zlim].max())
    ins.set_xlim(zlim)
    ins.set_yticklabels([])
    ins.set_ylabel('$p(z)$', fontsize=10)
    ins.set_xlabel('$z$', fontsize=10,  labelpad=-2.)
    # ins.tick_params(labelsize=12.5)
    ins.axvline(z_map[ez_idx], color='g', linestyle='--', zorder=10, lw=0.75)
    ins.axvspan(z_5, z_95, color='g', alpha=0.05, zorder=-1)
    ins.axvspan(z_16, z_84, color='g', alpha=0.25, zorder=-1)

    ins.text(0.95, 0.95, r'16%-68%', fontsize=4, transform=ins.transAxes, color='g',
            ha='right', va='top', alpha=0.5)
    ins.text(0.95, 0.8, r'5%-95%', fontsize=4, transform=ins.transAxes, color='g',
            ha='right', va='bottom', alpha=0.25)
    ins.text(0.95, 0.05, r'$z_{\rm MAP}$=%.2f'%(z_map[ez_idx]) + '\n' + r'$z_{\rm 50}$=%.2f'%(z_50_array[ez_idx]),
            fontsize=6, transform=ins.transAxes, color='k',  
            ha='right', va='bottom',  )
     
    
    # get cutout 
    ra, dec = ez.cat[ez_idx]['ra'], ez.cat[ez_idx]['dec']
    coord = SkyCoord(ra, dec, unit='deg')

    # show the cutout
    for i, (filt, img) in enumerate(image_dict.items()):
 
        ax = fig.add_subplot(gs[2 + i//10, i%10])
        
        cutout = Cutout2D(img.data, coord, 3*u.arcsec, wcs=img.wcs)
        
        valid = np.nan_to_num(cutout.data).sum()
        if valid == 0: font_color = 'k' ; font_ecolor = 'w'
        else:   font_color = 'w' ; font_ecolor = 'k'


        ax.imshow(cutout.data, origin='lower', cmap='binary_r', aspect='equal',
                norm=ImageNormalize(np.nan_to_num(cutout.data), interval=ZScaleInterval()))
        ax.text(0.05, 0.95, filt, fontsize=6, color=font_color, ha='left', va='top', transform=ax.transAxes,
                fontweight='bold', path_effects=[pe.withStroke(linewidth=1, foreground=font_ecolor)] )
        ax.set_xticks([])
        ax.set_yticks([])

        # add SNR
        mag = phot[phot['ID'] == id][f'{filt}_CIRC1'][0]
        err = phot[phot['ID'] == id][f'{filt}_CIRC1_e'][0]
        err = np.nanmax([phot[phot['ID'] == id][f'{filt}_CIRC1_en'][0], err])
    
        snr = mag / err
    
        ax.text(0.05, 0.05, 'S/N=%.1f' % (snr), fontsize=5., fontweight='bold',
                            color=font_color, ha='left', va='bottom', transform=ax.transAxes,
                            path_effects=[pe.withStroke(linewidth=1, foreground=font_ecolor)] )
    
    if save_dir:
        fig.savefig(os.path.join(save_dir, f'{save_suffix}_{id}.png'), dpi=200, bbox_inches='tight')
        plt.clf()
        plt.close(fig)
        return None
    else:
        return fig


if __name__ == "__main__":

 
    field = 'M0416'
    image_fname_func = lambda filt: f'/data/sapphires/all_mosaics/4750/4750_{filt.upper()}_v03_sci.fits'
    
    eazy_hdf5 ='/home/lxj/data/SAPPHIRE_EAZY/4750_v03/4750_v03_CIRC1.eazy.h5'
    phot_cat = '/data/sapphires/catalogs/4750_v03_merged_phot.fits'
    img_path = '/data/sapphires/all_mosaics/4750/'
    save_dir = '/home/lxj/data/SAPPHIRE_EAZY/4750_v03/plots/'
    save_suffix = 'SAPPIRES_4750_v03'

    phot_ver = 'v03'
    phot_type = 'CIRC1'


    ### ------- visualization ------- ###
 
    os.makedirs(save_dir, exist_ok=True)

    # load eazy
    ez = eazy.hdf5.initialize_from_hdf5(eazy_hdf5)

    # load photometry
    phot = Table.read(phot_cat)

    # load images
    image_list = {}
    for colname in phot.colnames:
        if colname.endswith('_CIRC1'):
            filt_tmp = colname.split('_')[0]
            tmp_img = image_fname_func(filt_tmp)
            if os.path.exists(tmp_img): image_list[filt_tmp] = tmp_img
            else: image_list[filt_tmp] = None
         
    print('Using images:', image_list)

    image_dict = {}
    for filt in image_list:
        if  image_list[filt] is None:
            print(f'{filt} image not found')
            continue
        index = 0
        try:
            img_data = fits.open(image_list[filt])[index].data
            img_header = fits.open(image_list[filt])[index].header
            image_dict[filt] = NDData(data=img_data, wcs=WCS(img_header))
        except:
            print(f'{filt} failed')
    

    # re-order the image_dict: F435W, F606W, F814W being the first three
    HST_list = ['F435W', 'F606W', 'F814W'] 
    HST_list = [filt for filt in HST_list if filt in image_dict]
    JWST_list = [filt for filt in image_dict if filt not in HST_list]
    JWST_list = sorted(JWST_list)
    image_dict = OrderedDict({filt: image_dict[filt] for filt in HST_list + JWST_list})
    print('Re-ordered images:', image_dict.keys())


    zlimits = ez.pz_percentiles(percentiles=[2.5, 5,16,50,84,95, 97.5],
                                          oversample=1)
    z_2p5_array = zlimits[:,0]
    z_5_array = zlimits[:,1]
    z_16_array = zlimits[:,2]
    z_50_array = zlimits[:,3]
    z_84_array = zlimits[:,4]
    z_95_array = zlimits[:,5]
    z_97p5_array = zlimits[:,6]

    idx_zmap = np.nanargmax(ez.lnp, axis=1)
    z_map = np.array([ez.zgrid[idx] for idx in idx_zmap])
    chi2_map = np.array([chi2_fit[idx] for chi2_fit, idx in zip(ez.chi2_fit, idx_zmap)])

    # # test
    # show_eazy_result(id=17114, save_dir='./', save_suffix='EAZY', 
    #                  fieldname='A2744', phot_ver='v1p0', phot_type='CIRC1')
    
    # visualization
    import multiprocessing
    from functools import partial
    os.makedirs(save_dir, exist_ok=True)
    show_ez = partial(show_eazy_result, fieldname=field, save_dir=save_dir, save_suffix=save_suffix,phot_ver=phot_ver, 
    phot_type=phot_type)
    with multiprocessing.Pool(20) as pool:
        pool.map(show_ez, phot['ID'])