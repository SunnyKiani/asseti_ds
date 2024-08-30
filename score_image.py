import asseti as ds
import json
import time
import numpy as np

from image_setup import image_setup
from implementation_add_tiledata_to_json import implementation_add_tiledata_to_json
from implementation_create_tiles import implementation_create_tiles
from score_tiles import score_tiles
from implementation_create_geo_df import implementation_create_geo_df
from implementation_simplify_geometries import implementation_simplify_geometries
from implementation_create_final_outputs import implementation_create_final_outputs
from create_results_image import create_results_image

def score_image(main_menu):
    # ---------------------------------------
    so = main_menu['score_options']
    # ---------------------------------------

    test = 0
    test_comment = ''
    if so['image_path'] is None:
        test_comment += "Must specify image file.\n"
        test += 1

    if so['output_path'] is None:
        test_comment += "Must specify output_path."
        test += 1

    # ---------------------------------------
    so = ds.dict_set_default(so, 'model', 'm2,m4,m7')
    so = ds.dict_set_default(so, 'device', 'cpu')
    so = ds.dict_set_default(so, 'rotateX', 0.0)
    so = ds.dict_set_default(so, 'rotateY', 0.0)
    so = ds.dict_set_default(so, 'rotateZ', 0.0)
    so = ds.dict_set_default(so, 'scaleX', 1.0)
    so = ds.dict_set_default(so, 'scaleY', 1.0)
    so = ds.dict_set_default(so, 'scaleZ', 1.0)
    so = ds.dict_set_default(so, 'translationX', 0.0)
    so = ds.dict_set_default(so, 'translationY', 0.0)
    so = ds.dict_set_default(so, 'translationZ', 0.0)
    so = ds.dict_set_default(so, 'score_batch_size', 25)
    so = ds.dict_set_default(so, 'oversample', 20)
    so = ds.dict_set_default(so, 'check', 0)
    # ---------------------------------------

    model_dict = {}
    model_dict['m1'] = [f'{ds.model_path}/m1_drone_cps_v4.pt']
    model_dict['m2'] = [f'{ds.model_path}/m2_drone_cps_v6.pt']
    model_dict['m3'] = [f'{ds.model_path}/m3_drone_ab_v1.pt']
    model_dict['m4'] = [f'{ds.model_path}/m4_drone_ab_v2.pt']
    model_dict['m5'] = [f'{ds.model_path}/m5_plane_abcps_v1.pt']
    model_dict['m6'] = [f'{ds.model_path}/m6_plane_abcps_v2.pt']
    model_dict['m7'] = [f'{ds.model_path}/m7_plane_abcps_v3.pt']

    # Create the modelpaths list (if using abbreviations or scoring multiple models at once)
    # The assumption here is that the model arg can either be a path to a model, e.g:
    # model=/some/path/to/model.pt
    # or a model abbreviation, such as:
    # model=v3d
    # or a comma-separate list of model abbreviations, such as:
    # model=v2,v3d
    model_str = so['model']
    model_str_list = model_str.split(',')
    modelpaths = []
    model_abbreviation = False
    for mod in model_str_list:
        if mod in model_dict.keys():
            modelpaths += model_dict[mod]
            model_abbreviation = True

    # ---------------------------------------

    if so['check'] == 1:
        print(json.dumps(so, indent=4, sort_keys=True))
        test += 1
    else:
        print(json.dumps(so, indent=4, sort_keys=True))

    if len(modelpaths) > 0:
        so['modelpath'] = modelpaths

    score_options = so
    main_menu['score_options'] = score_options

    if test > 0:
        print(test_comment)
    else:
        # ---------------------------------------
        # setup: create the directory and image headers
        section_title = 'stage 1/8 - creating the directory and image headers...'
        print(section_title)
        t1 = time.time()
        tstart = t1
        # print('-' * 20)
        # print('main_menu before image_setup:', main_menu)
        # print('-' * 20)

        main_menu = image_setup(main_menu)
        # print('main_menu after image_setup:', main_menu)
        # print('-' * 20)

        t2 = time.time()
        dt = np.round(t2 - t1, 2)
        msg = 'Time taken: ' + str(dt) + ' seconds\n'
        print(msg)

        # ---------------------------------------
        # add the tile data:
        section_title = 'stage 2/8 - adding the tile data to headers...'
        print(section_title)
        t1 = time.time()
        implementation_add_tiledata_to_json(main_menu, tx=1024, ty=1024, oversample_val=score_options['oversample'])
        t2 = time.time()
        dt = np.round(t2 - t1, 2)
        msg = 'Time taken: ' + str(dt) + ' seconds\n'
        print(msg)

        # ---------------------------------------
        # create the image tiles:
        section_title = 'stage 3/8 - creating the image tiles...'
        print(section_title)
        t1 = time.time()
        implementation_create_tiles(main_menu)
        t2 = time.time()
        dt = np.round(t2 - t1, 2)
        msg = 'Time taken: ' + str(dt) + ' seconds\n'
        print(msg)

        # ---------------------------------------
        # evaluate the model against the tiles:
        section_title = 'stage 4/8 - scoring the tiles...'
        print(section_title)
        t1 = time.time()
        score_tiles(main_menu)
        t2 = time.time()
        dt = np.round(t2 - t1, 2)
        msg = 'Time taken: ' + str(dt) + ' seconds\n'
        print(msg)

        # ---------------------------------------
        # Create geo_df (optional: create individual TILE labelmes)
        section_title = 'stage 5/8 - creating the geo dataframe...'
        print(section_title)
        t1 = time.time()
        implementation_create_geo_df(main_menu)
        t2 = time.time()
        dt = np.round(t2 - t1, 2)
        msg = 'Time taken: ' + str(dt) + ' seconds\n'
        print(msg)

        # ---------------------------------------
        # Create final outputs
        section_title = 'stage 6/8 - simplifying geometry points...'
        print(section_title)
        t1 = time.time()
        implementation_simplify_geometries(main_menu)
        t2 = time.time()
        dt = np.round(t2 - t1, 2)
        msg = 'Time taken: ' + str(dt) + ' seconds\n'
        print(msg)

        # ---------------------------------------
        # Create final outputs
        section_title = 'stage 7/8 - gathering the results...'
        print(section_title)
        t1 = time.time()
        implementation_create_final_outputs(main_menu)
        t2 = time.time()
        dt = np.round(t2 - t1, 2)
        msg = 'Time taken: ' + str(dt) + ' seconds\n'
        print(msg)

        # ---------------------------------------
        # Create output image

        section_title = 'stage 8/8 - creating output image...'
        print(section_title)
        t1 = time.time()
        create_results_image(main_menu)
        t2 = time.time()
        dt = np.round(t2 - t1, 2)
        msg = 'Time taken: ' + str(dt) + ' seconds\n'
        print(msg)

        # ---------------------------------------
        tend = t2
        dt = np.round(tend - tstart, 2)
        msg = 'TOTAL Time taken: ' + str(dt) + ' seconds\n'
        print(msg)

        print('Results saved in: ' + str(main_menu['new_imdir']))

#
# ==============================================================================
#
