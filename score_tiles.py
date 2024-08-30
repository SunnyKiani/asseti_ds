import asseti as ds
import os
import json
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from ultralytics import YOLO


def score_tiles(main_menu):
    parent_pkl = main_menu['master_pkl']
    parent_meta_data = ds.load_pickle(parent_pkl)
    parent_impath = parent_meta_data['imname']
    parent_hdr = parent_meta_data['hdr']
    score_opts = parent_meta_data['score_options']
    parent_fpath, parent_fname = os.path.split(parent_impath)

    score_opts = main_menu['score_options']
    parent_dir = score_opts['new_imdir']
    modelpaths = score_opts['modelpath']
    # print("Sunny - Paul, modelpaths:", modelpaths)
    score_device = score_opts['device']

    if not type(modelpaths) is list:
        modelpaths = [modelpaths]

    for modelpath in modelpaths:
        model_parentpath, model_fname = os.path.split(modelpath)
        model_suffix = os.path.splitext(model_fname)[-1]
        model_label = model_fname.replace(model_suffix, '')

        f_results_dir = f'{parent_dir}/{model_label}/results'
        f_master_result = f'{f_results_dir}.pkl'
        print(' ----- f_master_result:', f_master_result)
        ds.mkdir(f_results_dir)

        if not os.path.exists(f_master_result):
            im_glob = f'{parent_dir}/tiles/*.png'
            im_list = ds.glob_files(im_glob)

            # split the file list into chunks:
            nf = score_opts['score_batch_size']
            master_result = []
            bn = 1
            model = YOLO(modelpath)
            if score_device in ['0', '0,1,2,3']:
                model.to('cuda')

            while len(im_list) > 0:
                print(f'scoring batch: {bn}')
                f_results_file = f'{f_results_dir}/batch_results_{bn:04}.pkl'
                print('-------------------Sunny/ f_result_file:', f_results_file)
                f_chunk = im_list[:nf]
                im_list = im_list[nf:]
                if not os.path.exists(f_results_file):
                    results = model(f_chunk, device=score_device, verbose=False, stream=True)
                    try:
                        results = process_model_results(results)
                    except:
                        model = None
                        results = None
                        model = YOLO(modelpath)
                        results = model(f_chunk, device=score_device, verbose=False, stream=True)  # IMPORTANT <-- Sunny
                        results = process_model_results(results)
                    ds.save_pickle(results, f_results_file, compress=False)
                bn += 1

            f_results_glob = f'{f_results_dir}/batch_results*.pkl'
            f_results_files = ds.glob_files(f_results_glob)
            master_result = []
            for f_r in f_results_files:
                results = ds.load_pickle(f_r)
                master_result += results

            # print(master_result)

            ds.save_pickle(master_result, f_master_result, compress=False)
            print('--------------- master results:------------------')
            print('=============f_master_result:', f_master_result)
            # for i in master_result:
            #     print(i)
            #     print('---------------------')


#====================================================
def process_model_results(results):

    master_result = []
    for res in results: # One result per image in im_list
        # print('*' * 100)
        # print('confidence', res.boxes.conf)
        ncomps        = len(res)
        impath        = res.path
        im_suffix     = os.path.splitext(impath)[-1]
        pk1           = impath.replace(im_suffix,'.pkl')
        pk1           = pk1.replace('tiles','tiles_meta')
        f_components  = pk1.replace('.pkl','_components.json')
        meta_data     = ds.load_pickle(pk1)
        # print('*****meta****'*10)
        # print(meta_data)
        hdr           = meta_data['hdr']
        fpath, fname  = os.path.split(impath)
        f_labelme     = impath.replace(im_suffix, '.json')
        f_id          = fname.replace(im_suffix, '')

        a = {}
        a['f_id']         = f_id
        a['ncomps']       = ncomps
        a['imname']       = impath
        a['im_suffix']    = im_suffix
        a['pk1']          = pk1
        a['f_components'] = f_components
        a['fpath']        = fpath
        a['fname']        = fname
        a['f_labelme']    = f_labelme
        a['hdr']          = hdr
        a['components']   = []

        cn = 0
        for comp in res: # one comp per component in the image
            # print('===comp<>json==='*10)
            # print('comp.boxes:', comp.boxes)
            # print(comp)
            c_id = f'{cn:04}'
            dd   = json.loads(comp.tojson())

            model_class = dd[0]['name']

            obj_label = 'unknown'
            if model_class == 'Steel_Colourbond':
                obj_label = 'Exterior - Roof - Sheeting - Colorbond'
            if model_class == 'Polycarbonate':
                obj_label = 'Exterior - Roof - Sheeting - Polycarbonate'
            if model_class == 'Asset_Boundary':
                obj_label = 'Asset Boundary'

            xx = np.array(dd[0]['segments']['x'])
            yy = np.array(dd[0]['segments']['y'])
            # convert to LAT-LON
            pp,qq = ds.pixels2coords_hdr(xx,yy,hdr)

            b = {}
            b['c_id'] = c_id
            b['polygon_valid'] = 'T'
            b['model_class'] = model_class
            b['obj_label']   = obj_label
            b['obj_conf']    = dd[0]['confidence'] # Sunny

            if len(xx) > 2:
                polygon_pixels = Polygon(zip(xx,yy))
                if polygon_pixels.is_valid:
                    b['polygon_pixels'] = polygon_pixels
                else:
                    b['polygon_valid'] = 'F'
                    b['polygon_pixels'] = polygon_pixels.buffer(0)

                polygon_coords = Polygon(zip(pp,qq))
                if polygon_coords.is_valid:
                    b['polygon_coords'] = polygon_coords
                else:
                    b['polygon_valid'] = 'F'
                    b['polygon_coords'] = polygon_coords.buffer(0)

                b['df'] = pd.DataFrame({'x':xx, 'y':yy, 'Lon':pp, 'Lat':qq})

                a['components'] += [b]

            cn += 1

        master_result += [a]

    # print('---- for i in master_result::')
    # print(master_result)

    return master_result