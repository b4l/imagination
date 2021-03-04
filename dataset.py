import utils
from rasterio.plot import reshape_as_image
import matplotlib.pyplot as plt
from pathlib import Path
import geopandas as gpd
from multiprocessing import Pool

gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'

# list tiles and extract extent
tiles = gpd.GeoDataFrame.from_dict(
    {"path": list(Path("/data/swissimage").glob("*.tif"))})
tiles.geometry = tiles.path.apply(lambda x: utils.extent_from_filename(x))
tiles = tiles.set_crs(2056)
tiles.head()

# extract samples
labels = gpd.read_file("data/road_type_samples.kml", driver='KML')
labels.columns = labels.columns.str.lower()
labels = labels[["name", "geometry"]].to_crs(2056)
labels.geometry = labels.geometry.apply(lambda x: utils.extent_from_point(x))
labels["path"] = labels.name.apply(
    lambda x: Path("data/labels/{}.jpeg".format(x)))
labels.apply(lambda x: utils.extract_subset_file(tiles, x.geometry, x.path),
             axis=1)

labels["data"] = labels.geometry.apply(
    lambda x: reshape_as_image(utils.extract_subset(tiles, x)))

f, ax = plt.subplots(3, 3, figsize=(20, 20))
ax[0, 0].imshow(labels.data.iloc[0])
ax[0, 1].imshow(labels.data.iloc[1])
ax[0, 2].imshow(labels.data.iloc[2])
ax[1, 0].imshow(labels.data.iloc[3])
ax[1, 1].imshow(labels.data.iloc[4])
ax[1, 2].imshow(labels.data.iloc[5])
ax[2, 0].imshow(labels.data.iloc[6])
ax[2, 1].imshow(labels.data.iloc[7])
ax[2, 2].imshow(labels.data.iloc[8])

# extract images from points
images = gpd.read_file("data/road_points.geojson")
images.columns = images.columns.str.lower()
images = images[["id", "objektart", "geometry"]]
images = images[images.objektart != 11].append(
    images[images.objektart == 11].iloc[::10, :])
images.geometry = images.geometry.apply(lambda x: utils.extent_from_point(x))
images = images[images.geometry.within(tiles.dissolve().geometry.iloc[0])]
images["path"] = images.id.apply(lambda x: Path("data/img/{}.jpeg".format(x)))
images = images.reset_index()
images.head()

with Pool(None) as p:
    p.starmap(
        utils.extract_subset_file,
        zip([tiles] * images.shape[0], images.geometry.tolist(),
            images.path.tolist()))

print("Done!")
