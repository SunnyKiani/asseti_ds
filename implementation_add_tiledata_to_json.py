from asseti import load_pickle, save_pickle
import os
import numpy as np
from shapely.geometry import Polygon
import geopandas as gpd


def implementation_add_tiledata_to_json(main_menu, tx = 640, ty = 640, oversample_val = 20):
    pk1 = main_menu['master_pkl']
    print('Sunny - pk1:', pk1)
    in_path, in_fname = os.path.split(pk1)
    meta_data = load_pickle(pk1)
    fim  = meta_data['imname']
    im_suffix = os.path.splitext(fim)[-1]
    hdr  = meta_data['hdr']
    nx = hdr['naxis1']
    ny = hdr['naxis2']

    cx = nx // 2
    cy = ny // 2

    if oversample_val == -1:
        mx = np.floor(nx/tx)
        my = np.floor(ny/ty)
        nnx = mx*tx
        nny = my*ty
        xi = (tx * np.arange(mx)).astype('int')
        yi = (ty * np.arange(my)).astype('int')
        xgap = (nx - nnx).astype('int')
        xi = xi + (xgap // 2)
        ygap = (ny - nny).astype('int')
        yi = yi + (ygap // 2)


    if oversample_val > -1:
        mx = np.floor((1 + oversample_val/100) * nx/tx) + 1
        my = np.floor((1 + oversample_val/100) * ny/ty) + 1
        dx = np.floor((nx-tx)/(mx-1))
        dy = np.floor((ny-ty)/(my-1))
        xi = (dx * np.arange(mx)).astype('int')
        yi = (dy * np.arange(my)).astype('int')


    cell_names = []
    cell_polygons = []
    for j in range(len(yi)):
        for i in range(len(xi)):
            x1 = int(xi[i])
            x2 = int(x1 + tx)
            y1 = int(yi[j])
            y2 = int(y1 + ty)
            cell_name = 'cell_' + f'{i:02}' + '_' + f'{j:02}'
            cell_names += [cell_name]
            cell_polygon = Polygon([(x1, y1), (x1, y2), (x2, y2), (x2, y1), (x1, y1)])
            cell_polygons += [cell_polygon]

    cell_geo = {'name': cell_names, 'geometry': cell_polygons}
    cell_geo = gpd.GeoDataFrame(cell_geo)

    meta_data['tile_data'] = {}
    meta_data['tile_data']['tx'] = tx
    meta_data['tile_data']['ty'] = ty
    meta_data['tile_data']['xi'] = xi
    meta_data['tile_data']['yi'] = yi
    meta_data['tile_data']['cell_geo'] = cell_geo
    save_pickle(meta_data, pk1, compress=False)