import os
import cv2
import math
import json
import glob

from scipy.optimize import least_squares
import numpy as np
import pandas as pd

from pandasql import sqldf

import pickle
import bz2
import pickle as cPickle

import platform
plat = platform.platform()
if not 'arm64' in plat:
    from osgeo import gdal

from shapely import wkt
from shapely.geometry import Polygon
from shapely.ops import transform
import geopandas as gpd

import re
from PIL import Image
import base64
from pathlib import Path

pd.set_option('mode.chained_assignment', None)

# ==============================================================================
script_path = Path(__file__).resolve()
script_dir, script_name = os.path.split(script_path)
comp_path  = Path(__file__).resolve().parent.parent
repo_path  = Path(__file__).resolve().parent.parent.parent
model_path = os.path.join(comp_path, 'models')
model_class_mappings = os.path.join(script_dir, 'component_mappings.csv')

# ==============================================================================
#
class ds_constants():
    def __init__(self):
        self.r_earth = 6.371 * 10**6
        self.deg2rad = 2 * math.pi / 360
        self.lat_1deg = self.deg2rad * self.r_earth

    def lon_1deg(self,lat):
        return ( self.lat_1deg * math.cos(lat * self.deg2rad) )

cn = ds_constants()
#
# ==============================================================================
#
def mkdir(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

# ==============================================
def yvals2d(nx=1, ny=1, image=None):
    if not image is None:
        (ny,nx) = image.shape[:2]
    return np.arange(ny).reshape(ny,1) * np.ones(nx*ny).reshape(ny,nx)
#
# ==============================================================================
#
def load_json(fname):
    f = open (fname, "r")
    # Reading from file
    data = json.loads(f.read())
    # Closing file
    f.close()
    return data
#
# ==============================================================================
#
def glob_files(fpat):
    ff = glob.glob(fpat)
    ff.sort()
    return ff
#
# ==============================================================================
#
def save_textfile(data, outfile, mode='overwrite'):
    open_type = 'w'
    if mode == 'append':
        open_type = 'a'
    with open(outfile, open_type) as f_txt:
        print(data, file=f_txt)

# ===============================================
# Currently doesn't support GEO-DF type
def dict2json(dicty, my_indent=4):
    geo_type_list = ['Polygon','MultiPolygon']
    # numpy arrays, panads dfs, and geopandas dfs must be treated separately:
    dct_types = {}
    geo_crs = {}
    for key in dicty.keys():
        val = dicty[key]
        dct_types[key] = 'simple'
        vtype = type(val)
        vtype_str = type(val).__name__
        #print(f'{key} : {vtype}')


        if vtype_str == 'OptimizeResult':
            dicty[key] = None


        if vtype == np.ndarray:
            dct_types[key] = 'np_array'
            dicty[key] = val.tolist()


        if vtype == pd.core.frame.DataFrame:
            dct_types[key] = 'pd_df'
            # Sometimes Geo columns are hidden in regular data frames:
            geo_crs[key] = {}
            # Geo columns must be converted to strings:
            for col in val.columns:
                coltype = type(val[col])
                firstvaltype = type(val[col].iloc[0]).__name__
                is_geo = False
                if coltype == gpd.geoseries.GeoSeries:
                    is_geo = True
                    geo_crs[key][col] = val[col].crs
                if firstvaltype in geo_type_list:
                    is_geo = True
                    geo_crs[key][col] = None
                if is_geo:
                    dct_types[key] = 'gpd_gdf'
                    val[col] = val[col].apply(wkt.dumps)
            dicty[key] = val.to_dict()


        if vtype == gpd.geodataframe.GeoDataFrame:
            dct_types[key] = 'gpd_gdf'
            geo_crs[key] = {}
            # Geo columns must be converted to strings:
            for col in val.columns:
                coltype = type(val[col])
                firstvaltype = type(val[col].iloc[0]).__name__
                is_geo = False
                if coltype == gpd.geoseries.GeoSeries:
                    is_geo = True
                    geo_crs[key][col] = val[col].crs
                if firstvaltype in geo_type_list:
                    is_geo = True
                    geo_crs[key][col] = None
                if is_geo:
                    val[col] = val[col].apply(wkt.dumps)
            dicty[key] = val.to_dict()


        if vtype == gpd.geoseries.GeoSeries:
            dct_types[key] = 'gpd_series'
            geo_crs[key] = val.crs
            val = val.apply(wkt.dumps)
            dicty[key] = val.tolist()


        if vtype_str in geo_type_list:
            dct_types[key] = 'gpd_series'
            geo_crs[key] = None
            val = wkt.dumps(val)
            dicty[key] = [val]

    dicty['geo_crs'] = geo_crs
    dicty['types'] = dct_types

    # convert dictionary to JSON string
    json_string = json.dumps(dicty, indent=my_indent)

    return json_string
#
# ==============================================================================
#

def save_dict2json(dicty, f_js, my_indent=4):
    # convert dictionary to JSON string
    json_data = dict2json(dicty, my_indent)
    # write the JSON string to a file
    with open(f_js, 'w') as f:
        f.write(json_data)
#
# ==============================================================================
#
def determine_image_depth(im):
    max_vals = np.array([1,255,65535])
    img_max = im.max()
    delta = abs(max_vals - img_max)
    im_threshold = max_vals[np.where(delta == min(delta))][0]
    im_threshold = im_threshold.astype(np.float32) * 1.0
    return (im_threshold)
#
# ==============================================================================
#
def load_img(filename, switch_rgb=True, normalise=True,flip_y = True):
    im_suffix = os.path.splitext(filename)[-1]
    x = cv2.imread(filename, -1)
    if flip_y:
        x = np.flipud(x)
    x = x[:,:,0:3]
    if switch_rgb:
        x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    if normalise:
        col_max = determine_image_depth(x)
        x = x.astype(np.float32)/col_max
    return x
#
# ==============================================================================
#
def save_img(x, filename, switch_rgb=True, depth=16,flip_y = True):
    col_max = determine_image_depth(x)
    #print(col_max)
    im_suffix = os.path.splitext(filename)[-1]

    if im_suffix == '.png':
        if depth not in [8,16]:
            depth = 16
    if im_suffix == '.tif':
        if depth not in [8,16,32]:
            depth = 16
    if im_suffix in ['.jpg','.jpeg']:
        depth = 8

    if flip_y:
        x = np.flipud(x)

    x = x.astype(np.float32)/col_max
    if switch_rgb:
        x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)

    if depth == 8:
        x = (x*255).astype(np.uint8)
    if depth == 16:
        x = (x*65535).astype(np.uint16)
    if depth == 32:
        x = (x*1.0).astype(np.float32)
    cv2.imwrite(filename, x)
#
# ==============================================================================
#
# save a pickle:
def save_pickle(data, filename, compress=True):
    if compress:
        with bz2.BZ2File(filename + '.pbz2', 'w') as f:
            cPickle.dump(data, f)
    else:
        if os.path.exists(filename):
            com = f'rm {filename}'
            os.system(com)
        dbfile = open(filename, 'ab')
        pickle.dump(data, dbfile)
        dbfile.close()
#
# ==============================================================================
#
# load a pickle:
def load_pickle(filename):

    if filename[-4:] == 'pbz2':
        data = bz2.BZ2File(filename, 'rb')
        data = cPickle.load(data)
    else:
        dbfile = open(filename, 'rb')
        data = pickle.load(dbfile)
        dbfile.close()
    return data
#
# ==============================================================================
#

def read_textfile(outfile):
    with open(outfile, 'r') as f_txt:
        content = f_txt.read()
        f_txt.close()
    return content
#
# ==============================================================================
#
def get_coord_bounds(hdr):
    p1,q1 = pixels2coords_hdr(-0.5,-0.5, hdr)
    p2,q2 = pixels2coords_hdr(hdr['naxis1']-0.5,hdr['naxis2']-0.5, hdr)
    return p1,p2,q1,q2

#
# ==============================================================================
#
def get_buffered_coord_bounds(hdr,px_buffer=2):
    p1,q1 = pixels2coords_hdr(-0.5-px_buffer,-0.5-px_buffer, hdr)
    p2,q2 = pixels2coords_hdr(hdr['naxis1']-0.5+px_buffer,hdr['naxis2']-0.5+px_buffer, hdr)
    return p1,p2,q1,q2
#
# ==============================================================================
#
# R_z is the rotation about the z-axis in degrees:

def R_z(x,y,z,gamma):
    m = np.arange(9).reshape([3,3])*0
    cc = np.cos(gamma * np.pi/180)
    ss = np.sin(gamma * np.pi/180)

    x2 = x*cc -y*ss + 0
    y2 = x*ss +y*cc + 0
    z2 = 0 + 0 + 1*z
    return x2,y2,z2

# R_z is the rotation about the x-axis in degrees:

def R_x(x,y,z,alpha):
    m = np.arange(9).reshape([3,3])*0
    cc = np.cos(alpha * np.pi/180)
    ss = np.sin(alpha * np.pi/180)
    x2 = x
    y2 = 0 + y*cc -z*ss
    z2 = 0 + y*ss +z*cc
    return x2,y2,z2

# R_y is the rotation about the y-axis in degrees:

def R_y(x,y,z,beta):
    m = np.arange(9).reshape([3,3])*0
    cc = np.cos(beta * np.pi/180)
    ss = np.sin(beta * np.pi/180)
    x2 = x*cc + 0 + z*ss
    y2 = y
    z2 = -x*ss + 0 + z*cc
    return x2,y2,z2

# Fitting pitch and roll and yaw at the same time:
# This transformation is applied from closest to coordinate vector outwards
# Yaw is applied first so that it can be also considered in isolation
# PRY.C = Y.(R.(Y.C))

def calc_residual_complete(params, v1,v2):
    dx = params[0]
    dy = params[1]
    dz = params[2]
    alpha = -params[3]
    beta  = -params[4]
    gamma = -params[5]

    ll = int(len(v2)/3)
    x2 = v2[:ll] - dx
    y2 = v2[ll:2*ll] - dy
    z2 = v2[2*ll:] - dz

    x2,y2,z2 =  R_z(x2,y2,z2,gamma)
    x2,y2,z2 =  R_x(x2,y2,z2,alpha)
    x2,y2,z2 =  R_y(x2,y2,z2,beta)

    v2 = np.array(list(x2) + list(y2) + list(z2))

    res = v2-v1
    return res
#
# ==============================================================================
#

def get_coords_from_text(text):
    # Floats of kind 12.05 and integers
    regex = r"[\-]?\d+(\.\d+)?"
    numbers = []
    for match in re.finditer(regex, text):
        start = match.start()
        end = match.end()
        s = text[start:end]
        if "." in s:
            numbers.append(float(s))
        else:
            numbers.append(int(s))
    return numbers
#
# ==============================================================================
#
def get_dims_from_text(text):
    # Floats of kind 12.05 and integers
    regex = r" \d+x\d+ "
    numbers = []
    for match in re.finditer(regex, text):
        start = match.start()
        end = match.end()
        s = text[start:end]
        ss = s.split('x')
        for s in ss:
            numbers += [int(s)]
    return numbers
#
# ==============================================================================
#
# Create meaningful image headers:
def create_astro_hdr(fname,xoff=0,yoff=0):

    # If text header exists, use that, otherwise use GDAL:
    im_suffix = os.path.splitext(fname)[-1]
    f_txt = fname.replace(im_suffix,'.txt')
    if os.path.exists(f_txt):
        print("reading text file")
        aa = read_textfile(f_txt)
        print("read text file")
        aa = aa.split('\n')
        dummy_hdr = 0
        got_1 = 0
        got_2 = 0
        for a in aa:
            if 'Lower Left' in a:
                got_1 = 1
                tmp = get_coords_from_text(a)
                if len(tmp) == 4:
                    tmp = tmp[2:]
                p1,q1 = tmp

            if 'Upper Right' in a:
                got_2 = 1
                tmp = get_coords_from_text(a)
                if len(tmp) == 4:
                    tmp = tmp[2:]
                p2,q2 = tmp

            if '/image.' in a:
                nx,ny = get_dims_from_text(a)

        dummy_hdr = got_1*got_2
        if dummy_hdr == 0:
            # Normalise from 0-1
            p1 = 0.0
            q1 = 0.0
            p2 = 1.0
            q2 = 1.0

        cdelt1 = (p2-p1)/nx
        cdelt2 = (q2-q1)/ny
        crpix1 = nx/2
        crpix2 = ny/2
        crval1 = 0.5*(p1+p2)
        crval2 = 0.5*(q1+q2)
        naxis1 = nx
        naxis2 = ny
    else:
        print("opening with GDAL")
        gd = gdal.Open(fname)
        print("opened with GDAL")
        naxis1 = gd.RasterXSize
        naxis2 = gd.RasterYSize
        crval1,cdelt1,crpix1,crval2,crpix2,cdelt2 = gd.GetGeoTransform()


    # correct for the offsets:
    crval1 += xoff
    crval2 += yoff

    p1 = crval1 + (0 - crpix1)*cdelt1
    p2 = crval1 + (naxis1 - crpix1)*cdelt1
    q1 = crval2 + (0 - crpix2)*cdelt2
    q2 = crval2 + (naxis2 - crpix2)*cdelt2

    # Let's reference everything to the centre of the image:
    cx0 = naxis1 / 2
    cy0 = naxis2 / 2
    cp0 = 0.5*(p1 + p2)
    cq0 = 0.5*(q1 + q2)

    # I flip the image y-axes, so:
    cdelt2 = -cdelt2

    pix_dx = cn.deg2rad * cdelt1 * cn.r_earth
    pix_dy = cn.deg2rad * cdelt2 * cn.r_earth
    h = {}
    h['naxis1'] = naxis1
    h['naxis2'] = naxis2
    h['naxis3'] = 3
    h['cdelt1'] = cdelt1
    h['cdelt2'] = cdelt2
    h['cdelt3'] = 1
    h['crpix1'] = cx0
    h['crpix2'] = cy0
    h['crpix3'] = 0
    h['crval1'] = cp0
    h['crval2'] = cq0
    h['crval3'] = 0
    h['crota1'] = 0
    h['crota2'] = 0
    h['crota3'] = 0
    h['pix_dx'] = pix_dx
    h['pix_dy'] = pix_dy
    return h

#
# ==============================================================================
#
def slice_geo_image(im0, hdr0,x1,x2,y1,y2):
    im = im0[y1:y2,x1:x2,:].copy()
    ny0,nx0,nz0 = im0.shape
    cx0 = 0.5*(x1+x2)
    cy0 = 0.5*(y1+y2)
    # THIS LINE:
    cy0 = ny0 - cy0
    ny,nx,nz = im.shape
    cx = nx / 2
    cy = ny / 2
    pp0,qq0 = pixels2coords_hdr(cx0, cy0, hdr0)
    hdr = hdr0.copy()
    hdr['crval1'] = pp0
    hdr['crval2'] = qq0
    hdr['naxis1'] = nx
    hdr['naxis2'] = ny
    hdr['crpix1'] = cx
    hdr['crpix2'] = cy
    hdr['translationX'] = 0
    hdr['translationY'] = 0
    return im, hdr
#
# ==============================================================================
#
def jpeg2bytes(jpgfile):
    im_blob1 = open(jpgfile, "rb").read()
    im_blob2 = base64.b64encode(im_blob1)
    return im_blob2

def bytes2bytestring(bytestype):
    return bytestype.decode()

def bytestring2bytes(bytestring):
    return bytestring.encode()

def bytes2jpeg(bytestype, jpgfile='/tmp/tmp.jpg'):
    im_blob1 = base64.b64decode(bytestype)
    with open(jpgfile, 'wb') as f:
        f.write(im_blob1)

def bytes2np(bytestype):
    jpgfile='/tmp/tmp.jpg'
    bytes2jpeg(bytestype, jpgfile)
    pil_im = Image.open(jpgfile)
    numpy_array = np.array(pil_im)
    return numpy_array

def bytestring2np(bytestring):
    bytestype = bytestring2bytes(bytestring)
    return bytes2np(bytestype)

#
# ==============================================================================
#

def dedupe_geodf(geo,labelcol='geom_label',geomcol='geometry'):
    geo['dd_index'] = np.arange(len(geo))
    dd_geom = gpd.GeoSeries(geo[geomcol]).copy()

    geo['dd_area']  = dd_geom.area
    #print(geo['dd_area'].head())

    dd_rpoint = dd_geom.representative_point()
    dd_xy = dd_rpoint.get_coordinates(ignore_index=True)
    geo['dd_label'] = geo[labelcol]
    geo['dd_x']  = dd_xy.x
    geo['dd_y']  = dd_xy.y
    tmp = geo[['dd_index','dd_area','dd_x','dd_y','dd_label']].copy()
    n1 = len(tmp)
    tmp = sqldf('select dd_area,dd_x,dd_y,dd_label, min(dd_index) as dd_index from tmp group by dd_area,dd_x,dd_y,dd_label')
    n2 = len(tmp)
    nn = n1-n2
    print(f'Removed {nn} duplicate polygons.')
    tmp = tmp[['dd_index']].copy()
    geo = tmp.merge(geo, how='left', on='dd_index')
    geo = geo.sort_values(by=['dd_area'], ascending=[False])
    geo = geo.drop(columns=['dd_index', 'dd_area','dd_label','dd_x','dd_y'])
    return geo

#
# ==============================================================================
#
# SMART IMAGE DEFINITION
# A smart image consists of four parts:
# 1. A filename (imname) to an image file, usually jpeg
# 2. A header (hdr) for the imfile's coordinate mapping
# 3. A tall geo_dataframe (geo) with 'geometry_xy' column and 'geom_label' column
# (and probably, also a 'geometry_pq' column)
# 4. A metadata object for everything else (meta)

#
# ==============================================================================
#
def coords2pixels_hdr(pp,qq, hdr):
    if 'crota1' in hdr:
        cr1 = hdr['crota1']
    else:
        cr1 = 0.0
    cr1c = math.cos(cr1*math.pi/180)
    cr1s = math.sin(cr1*math.pi/180)
    p0   = hdr['crval1']
    q0   = hdr['crval2']
    x0   = hdr['crpix1']
    y0   = hdr['crpix2']
    cd1  = hdr['cdelt1']
    cd2  = hdr['cdelt2']
    dp   = pp - p0
    dq   = qq - q0
    xx   = x0 + (dp/cd1)*cr1c + (dq/cd2)*cr1s
    yy   = y0 - (dp/cd1)*cr1s + (dq/cd2)*cr1c
    return xx,yy

#
# ==============================================================================
#
def pixels2coords_hdr(xx,yy, hdr):
    if 'crota1' in hdr:
        cr1 = hdr['crota1']
    else:
        cr1 = 0.0
    cr1c = math.cos(cr1*math.pi/180)
    cr1s = math.sin(cr1*math.pi/180)
    p0   = hdr['crval1']
    q0   = hdr['crval2']
    x0   = hdr['crpix1']
    y0   = hdr['crpix2']
    cd1  = hdr['cdelt1']
    cd2  = hdr['cdelt2']
    dx   = xx - x0
    dy   = yy - y0
    pp   = p0 + (dx*cd1)*cr1c - (dy*cd2)*cr1s
    qq   = q0 + (dx*cd1)*cr1s + (dy*cd2)*cr1c
    return pp,qq
#
# ==============================================================================
#
def consolidate_geoms_respecting_classes(geoms):
    class_list = geoms.model_class.unique()

    master_geom_df = None
    for cl in class_list:
        sub_df = geoms[geoms.model_class == cl]
        consolidation_counter = 1
        while consolidation_counter > 0:
            sub_df, consolidation_counter = consolidate_geoms(sub_df)

        master_geom_df = pd.concat([master_geom_df, sub_df])

    return master_geom_df

#
# ==============================================================================
#

def consolidate_geoms(geoms):
    consolidation_counter = 0
    new_geom_df = None
    while(len(geoms) > 0):
        aa = geoms[:1].copy()
        bb = geoms[1:].copy()
        ii = does_polygon_interesect_geoDF(aa,geoms)
        consolidation_counter += (np.sum(ii*1) - 1)

        # If .... not all true or false:

        # consolidate the geoms that overlap with each other, and set equal to the geometry of aa:
        aa.geometry = [geoms[ii].unary_union]

        # append the new geometry to the dataset:
        new_geom_df = pd.concat([new_geom_df, aa])

        # set geoms to the remaining objects:
        geoms = geoms[ii==False]

    return new_geom_df, consolidation_counter

#
# ==============================================================================
#

def does_polygon_interesect_geoDF(my_polygon, my_geoDF):
    inp, res = my_polygon.sindex.query(my_geoDF.geometry, predicate='intersects')
    has_intersect = np.isin(np.arange(0, len(my_geoDF)), inp)
    return has_intersect

#
# ==============================================================================
#
def unpack_geoms_object(geom_list):
    poly_list   = []
    while len(geom_list) > 0:
        this_geom = geom_list[0]
        geom_list = geom_list[1:]
        if this_geom.geom_type in ['Polygon','MultiPolygon','GeometryCollection']:
            if this_geom.geom_type == 'Polygon':
                poly_list += [this_geom]
            else:
                geom_list += this_geom.geoms

    return poly_list

#
# ==============================================================================
#

# ==============================================================================
#

def unpack_geoms(geo_df):
    new_geo = geo_df[:1].copy()
    new_geo = new_geo.drop(list(np.arange(len(new_geo))))
    #print(len(geo_df))
    for index, row in geo_df.iterrows():
        this_geom = row.geometry
        geoms = unpack_geoms_object([this_geom])
        for g in geoms:
            this_row = row.copy()
            this_row.geometry = g
            ii = len(new_geo)
            new_geo.loc[ii] = this_row
    return new_geo
#
# ==============================================================================
#

def simplify_geo_df(geo_df):
    new_geo = geo_df.copy()
    classes = new_geo.model_class.unique()
    new_geo = new_geo.drop(list(np.arange(len(new_geo))))

    for cl in classes:
        this_df = geo_df.copy()
        this_df = this_df[this_df.model_class == cl]
        while len(this_df) > 0:
            i = this_df.index[0]
            row = this_df.loc[i]
            this_poly  = row.geometry
            this_type  = row.type
            this_class = row.model_class
            overlap_test  = this_df.overlaps(this_poly)
            ii = list((overlap_test[overlap_test == True]).index)
            ii += [i]
            select_rows = this_df.loc[ii]
            this_df = this_df.drop(ii)
            # remove the empty polygons:
            select_rows = select_rows[~(select_rows.is_empty)]
            if select_rows.shape[0] > 0:
                new_poly = select_rows.unary_union
                row.geometry = new_poly
                n = len(new_geo)
                new_geo.loc[n] = row

    new_geo = new_geo.reset_index(drop=True)
    return new_geo
#
# ==============================================================================
#

def calc_grad_intercept(p1,p2):
    gr = (p2[1]-p1[1])/(p2[0]-p1[0])
    y0 = p1[1] - gr*p1[0]
    return gr, y0
# ---------------------

def get_corners(xy):
    # start get corners
    minx = xy.x.min()
    maxx = xy.x.max()
    miny = xy.y.min()
    maxy = xy.y.max()
    boxdx = maxx - minx
    boxdy = maxy - miny
    x0   = xy.x.mean()
    y0   = xy.y.mean()

    hr = {}
    hr['minx'] = minx
    hr['maxx'] = maxx
    hr['miny'] = miny
    hr['maxy'] = maxy
    hr['boxdx'] = boxdx
    hr['boxdy'] = boxdy
    hr['x0'] = x0
    hr['y0'] = y0

    xy['xv'] = 0
    xy['yv'] = 0
    xy['cv'] = 0

    xy = xy.reset_index(drop=True)
    xy.loc[xy.x == minx, 'xv'] = 1
    xy.loc[xy.x == maxx, 'xv'] = 2
    xy.loc[xy.y == miny, 'yv'] = 1
    xy.loc[xy.y == maxy, 'yv'] = 2

    xy.cv = xy.xv + xy.yv
    cc = xy[xy.cv > 0]

    df = cc.groupby(['xv','yv'])[['xv','yv','x','y']].mean().reset_index(drop=True)
    df.xv = np.array(df.xv).astype('int')
    df.yv = np.array(df.yv).astype('int')

    df['xfac'] = df.xv
    df['yfac'] = df.yv
    df.loc[df.xv == 2, 'xfac'] = 1
    df.loc[df.yv == 2, 'yfac'] = 1
    df['fac'] = df['xfac'] + df['yfac']
    tmp1 = df.groupby(['xv'])[['xv','yv','fac']].max().reset_index(drop=True)
    tmp1 = tmp1[tmp1.xv > 0]
    tmp2 = df.groupby(['yv'])[['xv','yv','fac']].max().reset_index(drop=True)
    tmp2 = tmp2[tmp2.yv > 0]
    tmp = pd.concat([tmp1,tmp2])
    df['tag']  = [str(int(row.xv))+str(int(row.yv))+str(int(row.fac)) for ind,row in df.iterrows()]
    tmp['tag'] = [str(int(row.xv))+str(int(row.yv))+str(int(row.fac)) for ind,row in tmp.iterrows()]
    tmp = tmp[['tag']]
    df = tmp.merge(df, how='left', on='tag')

    df['corner'] = ''
    df = df.sort_values(by=['x']).reset_index(drop=True)
    df['x_order'] = 1 + np.arange(4)
    df = df.sort_values(by=['y']).reset_index(drop=True)
    df['y_order'] = 1 + np.arange(4)
    df = df.reset_index(drop=True)
    df.loc[df.y_order <=2, 'corner'] += 'b'
    df.loc[df.y_order > 2, 'corner'] += 't'
    tmp = df.copy()
    tmp = tmp.groupby(['corner'])[['corner','x']].min().reset_index(drop=True).rename(columns={'x':'xmin'})
    df = df.merge(tmp, how='left')
    df.loc[df.x == df.xmin, 'corner'] += 'l'
    df.loc[df.x != df.xmin, 'corner'] += 'r'
    df = df[['corner','x','y']]

    xc_unique = df.groupby(['x','y'])[['x','y']].mean().reset_index(drop=True)
    nc = len(xc_unique)

    # end get corners
    return df,nc,hr

#=============================================

def estimate_mvp_rect(xy):

    df,nc,hr = get_corners(xy)

    my_dict = None
    if nc == 4:
        my_dict = {}
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y))
        gdf = gdf.dissolve().convex_hull

        bl = df[df.corner == 'bl']
        br = df[df.corner == 'br']
        tl = df[df.corner == 'tl']
        tr = df[df.corner == 'tr']

        bl_x = np.array(bl.x)
        br_x = np.array(br.x)
        tl_x = np.array(tl.x)
        tr_x = np.array(tr.x)
        bl_y = np.array(bl.y)
        br_y = np.array(br.y)
        tl_y = np.array(tl.y)
        tr_y = np.array(tr.y)

        # estimage gradients from BL:
        p1 = [bl_x,bl_y]
        p2 = [br_x,br_y]
        gr1, y01 = calc_grad_intercept(p1,p2)

        p2 = [tl_x,tl_y]
        gr2, y02 = calc_grad_intercept(p1,p2)

        # estimage gradients from TR:
        p1 = [tr_x,tr_y]
        p2 = [tl_x,tl_y]
        gr3, y03 = calc_grad_intercept(p1,p2)

        p2 = [br_x,br_y]
        gr4, y04 = calc_grad_intercept(p1,p2)

        gradients  = np.array([gr1,gr2,gr3,gr4]).reshape(4)
        intercepts = np.array([y01,y02,y03,y04]).reshape(4)

        marginfac = 0.1
        boxdx = hr['boxdx']
        boxdy = hr['boxdy']
        x0 = hr['x0']
        y0 = hr['y0']
        minx = hr['minx']
        maxx = hr['maxx']
        miny = hr['miny']
        maxy = hr['maxy']
        margin = np.min([marginfac*boxdx, marginfac*boxdy])
        my_dict['origin'] = [x0,y0]
        my_dict['df']     = df
        my_dict['gdf']    = gdf
        my_dict['gradients']  = gradients
        my_dict['intercepts'] = intercepts
        my_dict['xbox'] = [minx,maxx]
        my_dict['ybox'] = [miny,maxy]
        my_dict['xbox_margin'] = [minx - margin, maxx + margin]
        my_dict['ybox_margin'] = [miny - margin, maxy + margin]

    return my_dict

#=====================================================

def poly_metrics(poly1,poly2):
    dc = {}
    area1 = poly1.area
    area2 = poly2.area
    areas = np.array([area1,area2])
    intersect = poly1.intersection(poly2).area
    union = area1 + area2 - intersect
    iou = intersect / union
    dc['area1'] = area1
    dc['area2'] = area2
    dc['intersect'] = intersect
    dc['union'] = union
    dc['iou'] = iou
    pot_iou = areas.min()/areas.max()
    dc['pot_iou'] = pot_iou
    modified_iou = iou/pot_iou
    dc['modified_iou'] = modified_iou

    return dc

#
# ==============================================================================
#

def one_line(z, x, y):
    return z[0]*x + z[1] - y

def get_line_intersect(l1,l2):
    m1,c1 = l1
    m2,c2 = l2
    dm = m2-m1
    dc = c2-c1
    x = -dc/dm
    y = m1*x + c1
    return x,y



def fit_rect(xy, dict):
    # calculate initial memberships:
    xy['grp1'] = 0
    xy['grp2'] = 0
    xy['grp3'] = 0
    xy['grp4'] = 0

    c1,c2,c3,c4 = dict['intercepts']
    m1,m2,m3,m4 = dict['gradients']

    x = np.array(xy.x)
    y = np.array(xy.y)
    xy['dist1'] = np.abs(y - (m1*x + c1))
    xy['dist2'] = np.abs(y - (m2*x + c2))
    xy['dist3'] = np.abs(y - (m3*x + c3))
    xy['dist4'] = np.abs(y - (m4*x + c4))

    xy['min_dist'] = xy[['dist1', 'dist2', 'dist3','dist4']].min(axis=1)

    xy = xy.reset_index(drop=True)
    xy.loc[xy.dist1 == xy.min_dist, 'grp1'] = 1
    xy.loc[xy.dist2 == xy.min_dist, 'grp2'] = 1
    xy.loc[xy.dist3 == xy.min_dist, 'grp3'] = 1
    xy.loc[xy.dist4 == xy.min_dist, 'grp4'] = 1

    xy['tote'] = xy[['grp1', 'grp2', 'grp3','grp4']].sum(axis=1)
    z_guess = np.array([m1,c1,m2,c2,m3,c3,m4,c4]).reshape(8)


    fit_m = []
    fit_c = []
    for i in range(4):
        j = i+1
        grp = f'grp{j}'
        xy_oneline = xy[xy[grp] == 1]
        i1 = 2*i
        i2 = 2*j
        z_guess_oneline = z_guess[i1:i2]
        res_lsq_oneline = least_squares(one_line, z_guess_oneline, loss='soft_l1', args=(xy_oneline.x, xy_oneline.y))
        key = f'line_{j}'
        #rect_fits[key] = res_lsq_oneline
        fit_m += [res_lsq_oneline.x[0]]
        fit_c += [res_lsq_oneline.x[1]]


    c1,c2,c3,c4 = fit_c
    m1,m2,m3,m4 = fit_m
    l1 = [m1,c1]
    l2 = [m2,c2]
    l3 = [m3,c3]
    l4 = [m4,c4]

    rect_fits = {}
    rect_fits['l1'] = l1
    rect_fits['l2'] = l2
    rect_fits['l3'] = l3
    rect_fits['l4'] = l4
    rect_fits['fit_c'] = fit_c
    rect_fits['fit_m'] = fit_m

    x_bl,y_bl = get_line_intersect(l1,l2)
    x_br,y_br = get_line_intersect(l1,l4)
    x_tl,y_tl = get_line_intersect(l3,l2)
    x_tr,y_tr = get_line_intersect(l3,l4)

    rect_fit_poly = Polygon([(x_bl, y_bl), (x_tl, y_tl), (x_tr, y_tr), (x_br, y_br), (x_bl, y_bl)])
    rect_fits['rect_fit_poly'] = rect_fit_poly
    return xy, rect_fits

#
# ==============================================================================
#

def dict_set_default(dict,key,dval):
    if key not in dict:
        dict[key] = dval
    return dict

#
# ==============================================================================
#

def rescale_imagefile(f_in, f_out, outx, outy):
    com = f'convert {f_in} -resize {outx}x{outy} -gravity center -extent {outx}x{outy} -quality 90 {f_out}'
    os.system(com)

def rescale_header(hdr, outx, outy):
    hdr2 = hdr.copy()
    hdr2['naxis1'] = outx
    hdr2['naxis2'] = outy
    hdr2['cdelt1'] = hdr['cdelt1'] * hdr['naxis1'] / hdr2['naxis1']
    hdr2['cdelt1'] = hdr['cdelt2'] * hdr['naxis2'] / hdr2['naxis2']
    hdr2['crpix1'] = hdr['crpix1'] * hdr2['naxis1'] / hdr['naxis1']
    hdr2['crpix2'] = hdr['crpix2'] * hdr2['naxis2'] / hdr['naxis2']
    return hdr2
#
# ==============================================================================
#

def rescale_geoms(geoms_in, xscalefac, yscalefac):

    def scalegeoms(x,y,z=None):
        x1 = x * xscalefac
        y1 = y * yscalefac
        return (x1,y1)

    geoms_out =  [ transform(scalegeoms, g1) for g1 in geoms_in ]
    return geoms_out

