import asseti as ds
from shapely.ops import transform
import geopandas as gpd

def neat_results(f_pkl):
    f_geo = f_pkl.replace('image','geo_optimised_df')
    f_res = f_pkl.replace('image','results')

    meta     = ds.load_pickle(f_pkl)
    geoms_df = ds.load_pickle(f_geo)
    # print('----meta----')
    # print(meta)
    # print('----end of meta--------')

    f_im  = meta['imname']
    f_im_small  = f_res.replace('.pkl','_downsized.jpg')
    f_res_small = f_res.replace('.pkl','_downsized.pkl')

    hdr = meta['hdr']

    nx = hdr['naxis1']
    ny = hdr['naxis2']

    geoms_df['geometry'] = gpd.GeoSeries(geoms_df.simplified_geoms)
    geoms_df = ds.unpack_geoms(geoms_df)

    geoms_df['geometry_pq'] = gpd.GeoSeries(geoms_df.geometry)

    # convert from the LAT-LONGs to the XY:
    def c2pix_tfunc(x,y,z=None):
        x1,y1 = ds.coords2pixels_hdr(x,y, meta['hdr'])
        y1 = meta['hdr']['naxis2'] - y1
        return (x1,y1)

    geometry_xy =  [ transform(c2pix_tfunc, g1) for g1 in geoms_df.geometry_pq ]
    geoms_df['geometry_xy'] = gpd.GeoSeries(geometry_xy)
    geoms_df.to_csv('geoms_df.csv', index=False)

    geoms_df = geoms_df[['geometry_xy','geometry_pq','model_class','type']].copy()
    geoms_df['geom_xy_area'] = geoms_df.geometry_xy.area
    geoms_df['geom_pq_area'] = geoms_df.geometry_pq.area

    result = {}
    result['imfile'] = meta['imname']
    result['score_options'] = meta['score_options']
    result['hdr'] = meta['hdr']
    result['conf'] = meta['tile_data']['big_df_geo']['confidence']

    # now also create the readable results at a reasonable image size:
    outx = 2000
    xscalefac = outx/nx
    outy = int(round(xscalefac * ny))
    ds.rescale_imagefile(f_im, f_im_small, outx, outy)
    hdr2 = ds.rescale_header(result['hdr'], outx, outy)
    yscalefac = outy/ny
    thumbnail_xy = ds.rescale_geoms(geoms_df.geometry_xy, xscalefac, yscalefac)

    geoms_df['thumbnail_xy'] = thumbnail_xy

    #im_small_src = jpeg2bytes(f_im_small)
    #im_small_src = bytes2bytestring(im_small_src)

    result['thumbnail_imfile'] = f_im_small
    result['thumbnail_hdr'] = hdr2
    result['geoms_df'] = geoms_df

    ds.save_pickle(result, f_res, compress=False)