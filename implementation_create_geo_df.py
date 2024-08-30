import asseti as ds
import os
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import transform
import geopandas as gpd
import matplotlib.pyplot as plt

def implementation_create_geo_df(main_menu, save_tile_labelme = False):
    master_pkl = main_menu['master_pkl']
    tjs = f'{ds.script_dir}/labelme_template.json'
    d_labelme0     = ds.load_json(tjs)
    shape_template = d_labelme0['shapes'][0]

    # print('=====master_pkl======================')
    # print('master_pkl', master_pkl)
    parent_meta_data = ds.load_pickle(master_pkl)
    parent_impath    = parent_meta_data['imname']
    parent_hdr       = parent_meta_data['hdr']
    score_opts       = parent_meta_data['score_options']
    parent_dir       = score_opts['new_imdir']
    parent_fpath, parent_fname  = os.path.split(parent_impath)

    # convert from the LAT-LONGs to the XY:
    def c2pix_tfunc(x, y, z=None):
        x1, y1 = ds.coords2pixels_hdr(x, y, parent_hdr)
        y1 = parent_hdr['naxis2'] - y1
        return (x1, y1)

    score_opts = main_menu['score_options']
    modelpaths = score_opts['modelpath']


    for modelpath in modelpaths:
        model_parentpath, model_fname = os.path.split(modelpath)
        model_suffix = os.path.splitext(model_fname)[-1]
        model_label = model_fname.replace(model_suffix, '')

        f_results_dir = f'{parent_dir}/{model_label}/results'
        f_master_result = f'{f_results_dir}.pkl'

        f_geo_df = f'{parent_dir}/{model_label}/geo_df.pkl'
        f_geo_xy_df = f_geo_df.replace('.pkl', '_xy.pkl')
        f_model_geo = f'{parent_dir}/{model_label}/model_geo.pkl'

        # print("---- before the if condition, n =", n)
        if os.path.exists(f_model_geo):
            print('found: ' + str(f_model_geo))
            # print("---- if, n =", n)
        else:
            # print("---- else, n =", n)
            master_result = ds.load_pickle(f_master_result) # mo...r fu...r !!!

            master_polygons = []
            master_types = []
            master_classes = []
            master_confs = []

            # print('Sunny fffxxx-----------------')

            for res in master_result:  # One result per image in im_list
                # if res['components']:
                #     print('----res[components] exists! -------')
                #     for i in res:
                #         print('res - i:', i)
                #         print('res[i]:', res[i])
                #         print('='*20)

                # if res['components']

                f_id = res['f_id']
                ncomps = res['ncomps']
                # print('---- number of components, ncomps', ncomps)
                impath = res['imname']
                pk1 = res['pk1']
                hdr = res['hdr']
                fpath = res['fpath']
                fname = res['fname']

                im_suffix = res['im_suffix']
                f_labelme = res['f_labelme']
                f_components = res['f_components']

                d_labelme = d_labelme0.copy()
                cell_polygons = []
                cell_types = []
                cell_confs = []  # Sunny
                cell_classes = []
                cell_polygons_xy = []
                shapes = []
                if ncomps > 0:
                    for comp in res['components']:  # one comp per component in the image
                        if comp['polygon_valid'] == 'T':
                            obj_label = comp['obj_label']
                            model_class = comp['model_class']
                            obj_conf = comp['obj_conf'] # Sunny

                            shp = shape_template.copy()
                            shp['label'] = obj_label

                            x = np.array(comp['df']['x'])
                            y = np.array(comp['df']['y'])
                            p = np.array(comp['df']['Lon'])
                            q = np.array(comp['df']['Lat'])

                            cell_polygon = Polygon(zip(p, q))
                            cell_polygon_xy = Polygon(zip(x, hdr['naxis2'] - y))
                            if cell_polygon.is_valid:
                                cell_polygons += [cell_polygon]
                                cell_types += [obj_label]
                                cell_confs += [obj_conf] # Sunny
                                cell_classes += [model_class]
                                cell_polygons_xy += [cell_polygon_xy]

                            ptys = [[v[0], v[1]] for v in zip(x, y)]
                            shp['points'] = ptys
                            shapes += [shp]

                d_labelme['shapes'] = shapes

                if save_tile_labelme:
                    im1blob = open(impath, "rb")
                    im1blob = ds.base64.b64encode(im1blob.read()).decode()
                    d_labelme['imageData'] = im1blob
                    d_labelme['imagePath'] = fname
                    d_labelme['imageHeight'] = hdr['naxis2']
                    d_labelme['imageWidth'] = hdr['naxis1']
                    ds.save_dict2json(d_labelme, f_labelme)

                geo = gpd.GeoSeries(cell_polygons)
                geo_xy = gpd.GeoSeries(cell_polygons_xy)
                tile_geo_df = gpd.GeoDataFrame({'geometry': geo_xy})
                tile_geo_df['object_id'] = [f'obj_{i:04}' for i in range(len(cell_polygons_xy))]
                tile_geo_df['model_class'] = cell_classes

                # add the prediction geoms to the tile pikl for later visualisation
                tile_meta = ds.load_pickle(pk1)
                tile_meta[model_label] = {}
                tile_meta[model_label]['cell_polygons'] = cell_polygons
                tile_meta[model_label]['cell_types'] = cell_types
                tile_meta[model_label]['obj_conf'] = cell_confs # Sunny
                tile_meta[model_label]['cell_classes'] = cell_classes
                tile_meta[model_label]['tile_geo_df'] = tile_geo_df
                tile_meta['tile_data']['big_df_geo'] = tile_geo_df
                ds.save_pickle(tile_meta, pk1, compress=False)

                # print('---Sunny--------tile meta----------')
                # print('tile meta:', tile_meta) # Sunny

                master_polygons += cell_polygons
                master_types += cell_types
                master_classes += cell_classes
                master_confs += cell_confs # Sunny

            geo_df = gpd.GeoDataFrame({'geometry': master_polygons})
            geo_df['model_class'] = master_classes
            geo_df['type'] = master_types
            geo_df['conf'] = master_confs

            geo_df = geo_df.reset_index(drop=True)
            geo_df = ds.unpack_geoms(geo_df)
            # geo_df = simplify_geo_df(geo_df)
            # geo_df = simplify_geo_df(geo_df)
            geo_df['area'] = geo_df.geometry.area

            geo_df = ds.consolidate_geoms_respecting_classes(geo_df)

            geo_df_xy = geo_df.copy()

            '''
            nr = len(geo_df_xy)
            for i in range(nr):
                g1 = geo_df_xy.geometry.iloc[i]
                g2 = transform(c2pix_tfunc, g1)
                geo_df_xy.geometry.iloc[i] = g2
            '''
            for ind, row in geo_df_xy.iterrows():
                g1 = geo_df_xy.geometry.loc[ind]
                g2 = transform(c2pix_tfunc, g1)
                geo_df_xy.loc[ind, 'geometry'] = g2

            ds.save_pickle(geo_df, f_geo_df, compress=False)
            ds.save_pickle(geo_df_xy, f_geo_xy_df, compress=False)

            # Also save the polygons for later:
            model_geo = {}
            model_geo['master_polygons'] = master_polygons
            model_geo['master_types'] = master_types
            model_geo['master_classes'] = master_classes
            model_geo['master_confs'] = master_confs
            ds.save_pickle(model_geo, f_model_geo, compress=False)

            # end of for loop
            #-------------------------------------------------------

    print('Collating GEOs into Master...')
    # Now combine the potential multi-model results:
    grandmaster_polygons = []
    grandmaster_types = []
    grandmaster_classes = []
    grandmaster_confs = []
    for modelpath in modelpaths:
        model_parentpath, model_fname = os.path.split(modelpath)
        model_suffix = os.path.splitext(model_fname)[-1]
        model_label = model_fname.replace(model_suffix, '')

        f_model_geo = f'{parent_dir}/{model_label}/model_geo.pkl'

        model_geo = ds.load_pickle(f_model_geo)
        grandmaster_polygons += model_geo['master_polygons']
        grandmaster_types += model_geo['master_types']
        grandmaster_classes += model_geo['master_classes']
        grandmaster_confs += model_geo['master_confs']

    geo_df = gpd.GeoDataFrame({'geometry': grandmaster_polygons})
    geo_df['model_class'] = grandmaster_classes
    geo_df['type'] = grandmaster_types
    geo_df['confidence'] = grandmaster_confs

    print('unpacking the geometries...')
    geo_df = geo_df.reset_index(drop=True)
    geo_df = ds.unpack_geoms(geo_df)
    print('geoms unpacked.')

    geo_df['area'] = geo_df.geometry.area

    geo_df = ds.consolidate_geoms_respecting_classes(geo_df)

    combined_geo_df = f'{parent_dir}/geo_df.pkl'
    ds.save_pickle(geo_df, combined_geo_df, compress=False)

    # convert from the LAT-LONGs to the XY:
    def c2pix_tfunc(x, y, z=None):
        x1, y1 = ds.coords2pixels_hdr(x, y, parent_hdr)
        y1 = parent_hdr['naxis2'] - y1
        return (x1, y1)

    geo_df_xy = geo_df.copy()
    '''
    nr = len(geo_df_xy)
    for i in range(nr):
        g1 = geo_df_xy.geometry.iloc[i]
        g2 = transform(c2pix_tfunc, g1)
        geo_df_xy.geometry.iloc[i] = g2
    '''
    for ind, row in geo_df_xy.iterrows():
        g1 = geo_df_xy.geometry.loc[ind]
        g2 = transform(c2pix_tfunc, g1)
        geo_df_xy.loc[ind, 'geometry'] = g2

    combined_geo_df_xy = f'{parent_dir}/geo_df_xy.pkl'
    ds.save_pickle(geo_df_xy, combined_geo_df_xy, compress=False)
    # print('---Sunny: combined_geo_df_xy:', combined_geo_df_xy)
    # geo_df_xy.plot()
    # plt.savefig('geo_df_xy_plot.png')
    # plt.close()

    # geo_df.to_csv('geo_df_before_xy.csv', index=False)


    parent_meta_data['tile_data']['big_df_geo'] = geo_df_xy
    ds.save_pickle(parent_meta_data, master_pkl, compress=False)
    # print('===parent meta data==========')
    # print('parent_meta_data:', parent_meta_data)
