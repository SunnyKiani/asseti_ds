import asseti as ds
import os
import numpy as np

def implementation_create_tiles(main_menu):
    pk1 = main_menu['master_pkl']
    meta_data  = ds.load_pickle(pk1)
    fim        = meta_data['imname']
    hdr        = meta_data['hdr']
    score_opts = meta_data['score_options']
    cell_geo   = meta_data['tile_data']['cell_geo']
    cell_names = list(cell_geo.name)
    im0 = None

    in_path, in_fname = os.path.split(pk1)
    im_suffix  = os.path.splitext(fim)[-1]
    tile_path = score_opts['new_imdir'] + '/tiles'
    tile_path_meta = tile_path + '_meta'
    com = f'mkdir -p {tile_path} {tile_path_meta}'
    os.system(com)

    # tiles_done file:
    tiles_done = score_opts['new_imdir'] + '/tiles_done.txt'
    # print('Sunny - tile done dir:', tiles_done)
    if not os.path.exists(tiles_done):
        tx = meta_data['tile_data']['tx']
        ty = meta_data['tile_data']['ty']
        xi = meta_data['tile_data']['xi']
        yi = meta_data['tile_data']['yi']

        tc = 0
        for j in yi:
            for i in xi:
                x1 = int(i)
                x2 = int(x1 + tx)
                y1 = int(j)
                y2 = int(y1 + ty)
                str_ctr = f'{tc:04}'
                new_im_name = tile_path + '/' + in_fname.replace('.pkl','') + '_' + str_ctr + '.png'
                new_pkl_name = tile_path_meta + '/' + in_fname.replace('.pkl','') + '_' + str_ctr + '.pkl'

                if not os.path.exists(new_pkl_name):
                    if im0 is None:
                        im0 = ds.load_img(fim)
                    cell_name = cell_names[tc]
                    my_tile   = cell_geo[cell_geo.name == cell_name]

                    im1,hdr1 = ds.slice_geo_image(im0, hdr,x1,x2,y1,y2)

                    # Now we have a sliced image and header with associated object shapes:
                    new_meta = meta_data.copy()
                    new_meta['hdr'] = hdr1
                    new_meta['imname'] = new_im_name
                    i_y,i_x,i_z = im1.shape
                    i_test = i_y*i_x*i_z
                    if i_test == 1024*1024*3:
                        if np.sum(im1) > 0:
                            ds.save_img(im1,new_im_name)
                            ds.save_pickle(new_meta, new_pkl_name, compress=False)

                tc += 1

        # tiles done:
        com = f'touch {tiles_done}'
        os.system(com)

#
