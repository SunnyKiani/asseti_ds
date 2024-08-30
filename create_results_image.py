import asseti as ds
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon
import os

# Sunny:
from PIL import Image, ImageDraw, ImageFont

def create_results_image(main_menu):
    pk1 = main_menu['master_pkl']
    # print('----*'*20)
    # print('create_results_image:')
    # print(' Sunny pk1:', pk1)
    parent_meta_data = ds.load_pickle(pk1)
    score_opts       = parent_meta_data['score_options']
    parent_dir       = score_opts['new_imdir']
    neat_results_pkl = f'{parent_dir}/results.pkl'
    neat_results_image(neat_results_pkl)
#
# ==============================================================================

def neat_results_image(f_pkl, colour_dict=None):
    print('---------------Sunny, neat_results_image f_pkl:', f_pkl)

    result = ds.load_pickle(f_pkl)
    # print(result)

    # print('---------------result-------------------')
    # for i in result:
    #     print('i:', i)
    #     print('result[i]', result[i])
    #     # print('--columns:')
    #     # print('columns - result[i]:', type(i.columns))
    #     # print('-------')
    #     print('-----------end of result-----')
    # print('*******************************************')

    f_im = result['thumbnail_imfile']
    hdr  = result['thumbnail_hdr']
    geo  = result['geoms_df']
    geo.to_csv('geo_result.csv', index=False)


    # -------------------------------------
    # load the image stuff:
    img = ds.load_img(f_im)
    img_bytes = ds.jpeg2bytes(f_im)
    img_bytes = ds.bytes2bytestring(img_bytes)
    img_bytes = 'data:image/png;base64,' + img_bytes

    ny,nx,nz = img.shape

    # -------------------------------------
    # load the geometry stuff:
    geo.geometry =gpd.GeoSeries(geo.thumbnail_xy)
    geo = ds.unpack_geoms(geo)
    geo['geom_area'] = geo.geometry.area

    # -------------------------------------
    # create the dataset for the static html data cache
    html_template = 'image_template.html'
    html_str = ds.read_textfile(html_template)

    div_data = {}
    div_data['geometries'] = {}
    d_classes = []
    d_xcoords = []
    d_ycoords = []

    for ind,row in geo.iterrows():
        # print('==row=='*10)
        # print('row:', row)
        # print('-----the end-----------')
        d_classes += [row.model_class]
        xx,yy = row.geometry.exterior.coords.xy
        xx = list(np.array(xx))
        yy = list(np.array(yy))
        d_xcoords += [xx]
        d_ycoords += [yy]

    div_data['geometries']['class'] = d_classes
    div_data['geometries']['xcoords'] = d_xcoords
    div_data['geometries']['ycoords'] = d_ycoords

    div_data_json = ds.dict2json(div_data)
    #save_textfile(div_data_json, 'test.json')

    # -------------------------------------
    # Create the html content:
    min_opacity = 0.25
    max_opacity = 0.35
    geo['geom_area'] = geo.geometry.area
    geo['opacity'] = (geo.geom_area - geo.geom_area.min())/(geo.geom_area.max() - geo.geom_area.min())
    geo.opacity = min_opacity + (max_opacity-min_opacity)*(1-geo.opacity)
    geo = geo.sort_values(by=['geom_area'], ascending=[False])
    geo = geo.reset_index(drop=True)

    if colour_dict is None:
        # Sunny: uncomment and comment the next two lines concerning color mapping:
        # color_mapping = {"Polycarbonate": "magenta", "Steel_Colourbond": "blue","Asset_Boundary": "green"}
        color_mapping = {"Polycarbonate": "red", "Steel_Colourbond": "yellow", "Asset_Boundary": "cyan"}

        geo['colour']=geo['model_class'].map(color_mapping)

    my_svg_contents = ''
    n = 0
    for ind,row in geo.iterrows():
        n += 1
        # print('--------'*5)
        # print('row:', row)
        # print('---*'*10)
        xx,yy = row.geometry.exterior.coords.xy
        xx = list(np.array(xx))
        yy = list(ny -np.array(yy))
        cxy = [ f'{a[0]},{a[1]}' for a in zip(xx,yy) ]
        my_points = ' '.join(cxy)
        my_id = f'geom_{ind}'
        my_class = row.model_class
        z_index = ind
        fill_opacity = row.opacity
        fill_colour  = row.colour
        line_colour  = row.colour
        line_opacity = fill_opacity + 0.1
        my_style = f'stroke-opacity:{line_opacity}; stroke:{line_colour}; fill:{fill_colour}; fill-opacity:{fill_opacity}; z-index:{z_index}'
        this_polygon = f"<polygon id='{my_id}' class='{my_class}' points='{my_points}' style='{my_style}'/>\n"
        my_svg_contents += this_polygon

        #----------------------Sunny---------------------------
        # # Sunny: Calculate the centroid of the polygon
        polygon = Polygon(zip(xx, yy))
        centroid_x, centroid_y = polygon.centroid.x, polygon.centroid.y
        #
        # # Sunny: Add text element to display your name at the centroid of the polygon
        # this_text = f"<text x='{centroid_x}' y='{centroid_y}' font-family='Arial' font-size='14' fill='black'>Sunny</text>\n"
        #
        # my_svg_contents += this_polygon
        # my_svg_contents += this_text  # Add the text to the SVG content
        #-----------------------Sunny-----------------------------

        # # # Sunny: Add text element to display your name and confidence at the centroid of the polygon
        sunny_dirname = os.path.dirname(f_pkl)
        sunny_dir_geo_df = os.path.join(sunny_dirname, 'geo_df.pkl')
        # sunny_dir_geo_df = 'outputs//orth_blue-4.5cm/geo_df.pkl'

        geo_df = ds.load_pickle(sunny_dir_geo_df)
        sunny_length_geo_df = len(geo_df)
        if n <= sunny_length_geo_df: # 1111:
            confidence = result['conf'].iloc[ind] # Assuming confidence is stored in the row
            this_text = f"<text x='{centroid_x}' y='{centroid_y}' font-family='Arial' font-size='14' fill='black'> ({confidence:.2f})</text>\n"

            my_svg_contents += this_polygon
            my_svg_contents += this_text  # Add the text to the SVG content
        #--------------------------Sunny---------------------------------------------
        # Add text element to display the class label at the centroid of the polygon
        class_label = row['model_class']
        this_text = f"<text x='{centroid_x}' y='{centroid_y + 15}' font-family='Arial' font-size='14' fill='black'>{class_label}</text>\n"

        my_svg_contents += this_polygon
        my_svg_contents += this_text  # Add the text to the SVG content
        #-----------------------------Sunny-----------------------------------------------------


    print('=-----= n:', n)
    print('------------length geo_df:', sunny_length_geo_df)




    p_min = 0
    q_min = 0
    p_width = nx
    q_width = ny

    html_str = html_str.replace('canvas_width',str(nx))
    html_str = html_str.replace('canvas_height',str(ny))
    html_str = html_str.replace('p_min',str(p_min))
    html_str = html_str.replace('q_min',str(q_min))
    html_str = html_str.replace('p_width',str(p_width))
    html_str = html_str.replace('q_width',str(q_width))
    html_str = html_str.replace('my_byte_string',img_bytes)
    html_str = html_str.replace('my_data_cache_contents',div_data_json)
    html_str = html_str.replace('my_svg_contents',my_svg_contents)

    html_out = f_pkl.replace('.pkl','.html')
    # print("---------------Sunny html_str:", html_out)
    ds.save_textfile(html_str,html_out)
#
# ==============================================================================
#
