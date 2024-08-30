from asseti import mkdir, create_astro_hdr, save_dict2json, save_pickle
import os
import math

def image_setup(main_menu):
    score_options = main_menu['score_options']
    im_path = score_options['image_path']
    im_dir, im_fname = os.path.split(im_path)
    im_suffix = os.path.splitext(im_fname)[-1]
    # print('im_fname:', im_fname)
    # print('im_suffix:', im_suffix)
    im_fbasename = im_fname.replace(im_suffix,'')

    out_dir = score_options['output_path']
    mkdir(out_dir)

    # new imdir
    new_imdir  = f'{out_dir}/{im_fbasename}'
    main_menu['new_imdir'] = new_imdir
    mkdir(new_imdir)
    new_impath = f'{new_imdir}/image{im_suffix}'


    # copy image to out_dir
    if not os.path.exists(new_impath):
        com = f'cp {im_path} {new_impath}'
        os.system(com)

    f_js = new_impath.replace(im_suffix,'_hdr.json')
    master_pkl = new_impath.replace(im_suffix,'.pkl')
    # print('master_pkl:', master_pkl)

    f_txt = new_impath.replace(im_suffix,'.txt')
    #print(f_txt)
    com = f'identify -quiet {new_impath} > {f_txt}; listgeo -d {new_impath} >> {f_txt}'
    if not os.path.exists(f_txt):
        os.system(com)

    xoff = score_options['translationX'] * 180.0/ math.pi
    yoff = score_options['translationY'] * 180.0/ math.pi
    hdr  = create_astro_hdr(new_impath,xoff,yoff)
    hdr['crota1'] = 0.0
    hdr['crota2'] = 0.0
    hdr['crota3'] = 0.0

    # add some of the dir info:
    score_options['new_imdir'] = new_imdir

    # ---------------------------------------
    # create implementation pikls:
    master = {}
    master['imname'] = new_impath
    master['score_options'] = score_options
    master['hdr'] = hdr

    save_dict2json(master,f_js)
    save_pickle(master, master_pkl, compress=False)

    main_menu['master_pkl'] = master_pkl

    return main_menu

#
# ==============================================================================
#