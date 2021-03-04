import rasterio
import rasterio.merge
import rasterio.mask
from rasterio.windows import Window
from pathlib import Path
from shapely.geometry import box, Point


def extent_from_point(point, buffer=128):
    """Create extent from point with a buffer in pixels."""
    x = round(point.x, 1)  # align to pixel bounds
    y = round(point.y, 1)  # align to pixel bounds
    buffer_m = buffer * 0.1  # convert pixels to meters
    return box(x - buffer_m, y - buffer_m, x + buffer_m, y + buffer_m)


def extent_from_filename(path):
    """Create extent from swissimage file name."""
    parts = Path(path).stem.split("_")
    xy = list(map(lambda x: int(x) * 1000, parts[2].split("-")))
    return box(xy[0], xy[1], xy[0] + 1000, xy[1] + 1000)


def save_georeference_image(data, box, path):
    """Save an rasterio numpy array as georeference jpeg."""
    with rasterio.open(
            path,
            "w",
            driver="JPEG",
            height=256,
            width=256,
            count=3,
            dtype="uint8",
            crs=rasterio.crs.CRS.from_wkt(
                'LOCAL_CS["CH1903+ / LV95",UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","2056"]]'
            ),
            transform=rasterio.Affine(0.1, 0.0, round(box.bounds[0], 1), 0.0,
                                      -0.1, round(box.bounds[3], 1)),
    ) as dst:
        dst.write(data)


def sample_from_tiles(tiles, box, path=None):
    """Extract sample from swissimage tiles.

    If a path is given the images is saved as georeferenced jpeg.
    Return a rasterio numpy array.
    """
    x = round(box.bounds[0] + 0.5 * 0.1, 2)
    y = round(box.bounds[3] - 0.5 * 0.1, 2)

    path = tiles[tiles.intersects(Point(x, y))].path.iloc[0]
    src = rasterio.open(path)
    row, col = src.index(x, y)

    sample = None
    if row < 9744 and col < 9744:
        sample = src.read(window=Window(col, row, 256, 256))
    else:
        tiles_to_mosaic = tiles[tiles.intersects(box)]
        files_to_mosaic = tiles_to_mosaic.path.apply(
            lambda x: rasterio.open(x)).tolist()

        mosaic, _ = rasterio.merge.merge(files_to_mosaic)
        sample = mosaic[:, row:row + 256, col:col + 256]

    if path is not None and not Path(path).exists():
        save_georeference_image(sample, box, path)
    return sample


def sample_from_wms(box,
                    width=256,
                    height=256,
                    format="image/jpeg",
                    layers="ch.swisstopo.swissimage",
                    path=None):
    """Extract sample from swisstopo wms.

    If a path is given the images is saved as georeferenced jpeg.
    Return a rasterio numpy array.
    """
    url = "http://wms.geo.admin.ch/?SERVICE=WMS&REQUEST=GetMap&VERSION=1.3.0"
    url += "&LAYERS={}&STYLES=default".format(layers)
    url += "&CRS=EPSG:2056&BBOX={bbox}&WIDTH={w}&HEIGHT={h}&FORMAT={f}".format(
        bbox=",".join(map(str, map(round, box.bounds, [1] * 4))),
        w=width,
        h=height,
        f="image/jpeg")
    sample = rasterio.open(url).read()
    if path is not None and not Path(path).exists():
        save_georeference_image(sample, box, path)
    return sample
