import utils
import geopandas as gpd
import pandas as pd
import fiona

# region of intrest
boundaries = 'zip://geodata/ch.swisstopo.swissboundaries3d-gemeinde-flaeche.fill.zip'
boundaries += '!swissBOUNDARIES3D_1_3_LV95_LN02.gdb'
fiona.listlayers(boundaries)

boundaries = gpd.read_file(boundaries, layer='TLM_LANDESGEBIET')
boundaries.geometry = boundaries.simplify(10)
roi = boundaries.loc[boundaries.ICC == 'CH'].geometry.iloc[0]
del boundaries

# create random points
points = utils.generate_random_points(int(roi.area / (100 * 100)), roi)
gdf = gpd.GeoDataFrame.from_dict({'geometry': points}, geometry='geometry')
gdf.to_file('geodata/points.geojson', driver='GeoJSON')
# gdf = gpd.read_file('geodata/points.geojson')

# load streets
tlm_gdb = 'zip://geodata/swisstlm3d_2020-03_2056_5728.gdb.zip'
tlm_gdb += '!2020_SWISSTLM3D_FGDB101_CHLV95_LN02/SWISSTLM3D_CHLV95_LN02.gdb'
fiona.listlayers(tlm_gdb)

streets = gpd.read_file(tlm_gdb, layer='TLM_STRASSE')
streets.columns = streets.columns.str.lower()
streets = streets[(streets.stufe >= 0) & (streets.kunstbaute == 100)]
streets = streets[['uuid', 'objektart', 'geometry']]
streets.head()

# snap points to each road types
frames = []
for k in [0, 1, 2, 4, 5, 8, 9, 10, 11, 20, 21]:
    lines = streets[streets.objektart == k]

    # class 10 & 11 are ressource intensive, drop or lower the buffer
    lines_buffer = lines.buffer(10) if k in [10, 11] else lines.buffer(50)
    lines_buffer = gpd.GeoDataFrame(geometry=lines_buffer)
    selection = gpd.sjoin(gdf, lines_buffer, how='left').dropna().copy()

    line = lines.unary_union
    selection['geometry'] = selection.apply(
        lambda row: line.interpolate(line.project(row.geometry)), axis=1)

    selection['objektart'] = k
    selection = selection.drop(columns=['index_right'])
    frames.append(selection)

selections = pd.concat(frames).reset_index(drop=True)
selections["id"] = selections.index + 100000
selections.to_file('geodata/road_points.geojson', driver='GeoJSON')

# some post cleanup in QGIS
