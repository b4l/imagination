import utils
from rasterio.plot import reshape_as_image
import matplotlib.pyplot as plt
from pathlib import Path
import geopandas as gpd
from multiprocessing import Pool

gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'

# extract road type examples
examples = gpd.read_file("data/examples/road_types.kml", driver='KML')
examples.columns = examples.columns.str.lower()
examples = examples[["name", "geometry"]].to_crs(2056)
examples.geometry = examples.geometry.apply(
    lambda x: utils.extent_from_point(x))
examples["path"] = examples.name.apply(
    lambda x: Path("data/examples/{}.jpeg".format(x)))
examples["data"] = examples.geometry.apply(lambda x: utils.sample_from_wms(x))

# plot examples
plt.figure(figsize=(10, 10))
for i, row in examples.iterrows():
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(reshape_as_image(row.data))
    plt.title(row['name'])
    plt.axis("off")

# list tiles and extract extent
tiles = gpd.GeoDataFrame.from_dict(
    {"path": list(Path("/data/swissimage").glob("*.tif"))})
tiles.geometry = tiles.path.apply(lambda x: utils.extent_from_filename(x))
tiles = tiles.set_crs(2056)
tiles.head()

# extract street images from points
images = gpd.read_file("data/road_points.geojson")
images = images[images.objektart != 9].append(
    images[images.objektart == 9].iloc[::6, :])  # balance classes
images.geometry = images.geometry.apply(lambda x: utils.extent_from_point(x))
images = images[images.geometry.within(tiles.dissolve().geometry.iloc[0])]
images["path"] = images.apply(lambda x: Path("data/streets/{o}/{id}_{o}.jpeg".
                                             format(o=x.objektart, id=x.id)),
                              axis=1)
images.head()

with Pool(None) as p:
    p.starmap(
        utils.sample_from_tiles,
        zip([tiles] * images.shape[0], images.geometry.tolist(),
            images.path.tolist()))

print("Done!")
