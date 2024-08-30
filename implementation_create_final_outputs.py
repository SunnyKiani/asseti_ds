 #Now for the overall master_labelme and master_components:
import asseti as ds
from neat_results import neat_results
import os
import numpy as np
import geopandas as gpd
import base64

script_dir, script_name = os.path.split(ds.script_path)

def implementation_create_final_outputs(main_menu):
    master_pkl = main_menu['master_pkl']
    print('----***--- master_pkl final:', master_pkl)
    tjs = f'{script_dir}/labelme_template.json'
    d_labelme0     = ds.load_json(tjs)
    shape_template = d_labelme0['shapes'][0]

    parent_meta_data = ds.load_pickle(master_pkl)
    parent_impath    = parent_meta_data['imname']
    parent_hdr       = parent_meta_data['hdr']
    score_opts       = parent_meta_data['score_options']
    parent_dir       = score_opts['new_imdir']
    parent_fpath, parent_fname  = os.path.split(parent_impath)

    d_master_components = {}
    d_master_components['components'] = []
    d_master_labelme = d_labelme0.copy()
    f_master_labelme = parent_dir + '/image.json'
    f_master_comps   = parent_dir + '/components.json'
    print('----****---f_master_comps final:', f_master_comps)

    if not os.path.exists(f_master_labelme):
        f_geo_df = parent_dir + '/geo_df.pkl'
        f_optimised_geo     = f'{parent_dir}/geo_optimised_df.pkl'
        f_optimised_geo_xy  = f'{parent_dir}/geo_optimised_df_xy.pkl'
        geo_df   = ds.load_pickle(f_optimised_geo)

        geo_df = geo_df.reset_index(drop=True)

        geo_df['geom_type'] = geo_df.geometry.geom_type
        geo_df.geometry = gpd.GeoSeries(geo_df.simplified_geoms)
        #print(geo_df.groupby(['geom_type'])['geom_type'].size().reset_index(name='counts'))
        geo_df = ds.unpack_geoms(geo_df)
        geo_df['geom_type'] = geo_df.geometry.geom_type
        #print(geo_df.groupby(['geom_type'])['geom_type'].size().reset_index(name='counts'))

        # ptg - changing to simplified geometry:
        # master_geo   = geo_df.geometry
        master_geo   = geo_df.geometry
        master_types = list(geo_df['type'])
        master_confs = list(geo_df['confidence']) # Sunny

        shapes = []
        for i in range(len(master_geo)):
            this_type = master_types[i]
            this_conf = master_confs[i] # Sunny
            pq = master_geo[i].exterior.coords.xy
            pp = np.array(list(pq[0]))
            qq = np.array(list(pq[1]))
            dct = {}
            dct['id']     = ''
            dct['name']   = ''
            dct['type']   = this_type
            dct['conf']   = this_conf # Sunny
            dct['points'] = []
            shp = shape_template.copy()
            shp['label'] = this_type
            shp['conf'] = this_conf # Sunny

            xx,yy = ds.coords2pixels_hdr(pp,qq,parent_hdr)
            ptys  = [[x[0],x[1]] for x in zip(xx,yy)]
            shp['points'] = ptys
            shapes += [shp]

            alt = 100.0
            for i in range(len(pp)):
                pt = {}
                pt['Latitude']  = qq[i]
                pt['Longitude'] = pp[i]
                pt['Altitude']  = alt
                dct['points']  += [pt]

            d_master_components['components'] += [dct]

        d_master_labelme['shapes'] = shapes

        im1blob = open(parent_impath, "rb")
        im1blob = base64.b64encode(im1blob.read()).decode()
        d_master_labelme['imageData']   = im1blob
        d_master_labelme['imagePath']   = parent_fname
        d_master_labelme['imageHeight'] = parent_hdr['naxis2']
        d_master_labelme['imageWidth']  = parent_hdr['naxis1']

        ds.save_dict2json(d_master_labelme,    f_master_labelme)
        ds.save_dict2json(d_master_components, f_master_comps)

    # Now make the neat results:
    neat_results(master_pkl)
