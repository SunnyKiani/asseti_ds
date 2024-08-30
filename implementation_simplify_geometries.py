from operator import index

import asseti as ds
from shapely.validation import make_valid
from shapely import wkt
import numpy as np
from shapely.ops import transform

def implementation_simplify_geometries(main_menu):
    pk1 = main_menu['master_pkl']
    # print('-------Sunny--------pk1--------------------')
    # print('pk1:', pk1)
    # print('-----------------------')
    parent_meta_data = ds.load_pickle(pk1)
    # print('-----parentmeta data -----------------')
    # print('parent_meta_data:', parent_meta_data)
    # print('--------------------')
    parent_impath    = parent_meta_data['imname']
    parent_hdr       = parent_meta_data['hdr']
    score_opts       = parent_meta_data['score_options']
    parent_dir       = score_opts['new_imdir']
    f_master_geo     = f'{parent_dir}/geo_df.pkl'
    print('------------------f_master_geo:', f_master_geo)
    f_master_geo_xy  = f'{parent_dir}/geo_df_xy.pkl'

    f_optimised_geo     = f'{parent_dir}/geo_optimised_df.pkl'
    f_optimised_geo_xy  = f'{parent_dir}/geo_optimised_df_xy.pkl'

    geo_df = ds.load_pickle(f_master_geo)
    empty_polygon = wkt.loads('POLYGON EMPTY')

    mvp_rect_polys = []
    mvp_rect_ious = []
    mvp_rect_mod_ious = []
    mvp_rect_dicts = []
    rect_fit_ious = []
    rect_fit_mod_ious = []
    rect_fit_polys = []
    rect_fit_dicts = []

    minrect_polys = []
    minrect_ious = []
    minrect_mod_ious = []
    minrect_dicts = []

    simplex_polys = []
    simplex_ious = []
    simplex_mod_ious = []
    simplex_dicts = []

    ngeos = len(geo_df)
    for i in range(ngeos):
        do_rect=True

        if (i > 0) & (i % 100 == 0):
            print('Simplified ' + str(i) + ' of ' + str(ngeos) + ' geometries')
        this_poly = geo_df[i:i+1]
        poly_actual = this_poly.iloc[0].geometry

        # minrect
        minrect = poly_actual.minimum_rotated_rectangle
        my_metrics = ds.poly_metrics(minrect, poly_actual)
        minrect_ious += [my_metrics['iou']]
        minrect_mod_ious += [my_metrics['modified_iou']]
        minrect_polys += [minrect]
        minrect_dicts += [ds.dict2json(my_metrics)]

        # minrect
        simplex = poly_actual.simplify(0.000001, preserve_topology=True)
        my_metrics = ds.poly_metrics(simplex, poly_actual)
        simplex_ious += [my_metrics['iou']]
        simplex_mod_ious += [my_metrics['modified_iou']]
        simplex_polys += [simplex]
        simplex_dicts += [ds.dict2json(my_metrics)]


        xy = this_poly.geometry.get_coordinates()
        mvp_rect_dict = ds.estimate_mvp_rect(xy)
        if mvp_rect_dict is None:
            do_rect = False
        if do_rect == True:
            poly_mvp = mvp_rect_dict['gdf'].iloc[0]
            poly_mvp = make_valid(poly_mvp)

            mvp_rect_metrics = ds.poly_metrics(poly_mvp,poly_actual)
            poly_areas = np.array([poly_actual.area, poly_mvp.area])
            # calculate the maximum iou can be for these shapes:
            potential_iou = poly_areas.min()/poly_areas.max()
            # save the og iou and the modified iou:
            iou = mvp_rect_metrics['iou']
            modified_iou = iou/potential_iou
            mvp_rect_ious += [iou]
            mvp_rect_mod_ious += [modified_iou]

            mvp_rect_polys += [poly_mvp]
            mvp_rect_json = ds.dict2json(mvp_rect_dict)
            mvp_rect_dicts += [mvp_rect_json]

            # --------
            # now fit a rectangle to the mvp data:
            xy,rect_fit_dict = ds.fit_rect(xy, mvp_rect_dict)
            rect_fit_poly = rect_fit_dict['rect_fit_poly']
            rect_fit_poly = make_valid(rect_fit_poly)
            rect_fit_metrics = ds.poly_metrics(rect_fit_poly,poly_actual)
            poly_areas = np.array([poly_actual.area, rect_fit_poly.area])
            # calculate the maximum iou can be for these shapes:
            potential_iou = poly_areas.min()/poly_areas.max()
            # save the og iou and the modified iou:
            iou = rect_fit_metrics['iou']
            modified_iou = iou/potential_iou
            rect_fit_ious += [iou]
            rect_fit_mod_ious += [modified_iou]
            rect_fit_polys += [rect_fit_poly]
            rect_fit_dict_json = ds.dict2json(rect_fit_dict)
            rect_fit_dicts += [rect_fit_dict_json]
        else:
            mvp_rect_ious += [0.0]
            mvp_rect_mod_ious += [0.0]
            mvp_rect_polys += [empty_polygon]
            mvp_rect_dicts += ['']
            rect_fit_ious += [0.0]
            rect_fit_mod_ious += [0.0]
            rect_fit_polys += [empty_polygon]
            rect_fit_dicts += ['']

    geo_df['minrect_poly'] = minrect_polys
    geo_df['minrect_iou']  = minrect_ious
    geo_df['minrect_mod_iou'] = minrect_mod_ious
    geo_df['minrect_dict_json'] = minrect_dicts
    geo_df['simplex_poly'] = simplex_polys
    geo_df['simplex_iou']  = simplex_ious
    geo_df['simplex_mod_iou'] = simplex_mod_ious
    geo_df['simplex_dict_json'] = simplex_dicts
    geo_df['mvp_rect_poly'] = mvp_rect_polys
    geo_df['mvp_rect_iou']  = mvp_rect_ious
    geo_df['mvp_rect_mod_iou'] = mvp_rect_mod_ious
    geo_df['mvp_rect_dict_json'] = mvp_rect_dicts
    geo_df['rect_fit_poly'] = rect_fit_polys
    geo_df['rect_fit_iou']  = rect_fit_ious
    geo_df['rect_fit_mod_iou'] = rect_fit_mod_ious
    geo_df['rect_fit_dict_json'] = rect_fit_dicts

    # ----------------------------------------------------------
    # Any additional improvements to simplified_geoms goes here:
    # ----------------------------------------------------------
    # Improvements e.g. for more general polygons (triangle, hexagon)
    geo_df['simplified_geoms'] = minrect_polys
    geo_df['simplified_geoms_iou'] = minrect_ious
    geo_df['chosen_geometry'] = 'minrect'

    geo_df = geo_df.reset_index(drop=True)


    ii = (geo_df.simplex_iou > geo_df.simplified_geoms_iou)
    geo_df.loc[ii, 'simplified_geoms'] = geo_df.loc[ii, 'simplex_poly']
    geo_df.loc[ii, 'simplified_geoms_iou'] = geo_df.loc[ii, 'simplex_iou']
    geo_df.loc[ii, 'chosen_geometry'] = 'simplex'

    geo_df_columns = geo_df.columns

    iname = 'mvp_rect_iou'
    gname = 'mvp_rect_poly'
    if iname in geo_df_columns:
        ii = (geo_df[iname] > geo_df.simplified_geoms_iou)
        geo_df.loc[ii, 'simplified_geoms'] = geo_df.loc[ii, gname]
        geo_df.loc[ii, 'simplified_geoms_iou'] = geo_df.loc[ii, iname]
        geo_df.loc[ii, 'chosen_geometry'] = 'mvp_rect'

    iname = 'rect_fit_iou'
    gname = 'rect_fit_poly'
    if iname in geo_df_columns:
        ii = (geo_df[iname] > geo_df.simplified_geoms_iou)
        geo_df.loc[ii, 'simplified_geoms'] = geo_df.loc[ii, gname]
        geo_df.loc[ii, 'simplified_geoms_iou'] = geo_df.loc[ii, iname]
        geo_df.loc[ii, 'chosen_geometry'] = 'fit_rect'

    # Create a threshold here based upon modified iou:

    #
    #
    #
    # ----------------------------------------------------------


    ds.save_pickle(geo_df, f_optimised_geo, compress=False)
    print('------------foooo------------')
    print('f_optimised_geo:', f_optimised_geo)

    # convert from the LAT-LONGs to the XY:
    def c2pix_tfunc(x,y,z=None):
        x1,y1 = ds.coords2pixels_hdr(x,y, parent_hdr)
        y1 = parent_hdr['naxis2'] - y1
        return (x1,y1)

    geo_df_xy = geo_df.copy()
    for ind, row in geo_df_xy.iterrows():
        g1 = geo_df_xy.simplified_geoms.loc[ind]
        g2 = transform(c2pix_tfunc, g1)
        geo_df_xy.loc[ind, 'simplified_geoms'] = g2


    print('------------fuooooo X Y ------------')
    print('f_optimised_geo_xy:', f_optimised_geo)
    ds.save_pickle(geo_df_xy, f_optimised_geo_xy, compress=False)