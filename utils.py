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


def extract_subset(tiles, box):
    """Extract np array from swissimage tiles."""
    x = round(box.bounds[0] + 0.5 * 0.1, 2)
    y = round(box.bounds[3] - 0.5 * 0.1, 2)

    path = tiles[tiles.intersects(Point(x, y))].path.iloc[0]
    src = rasterio.open(path)
    row, col = src.index(x, y)

    if row < 9743 and col < 9743:
        return src.read(window=Window(col, row, 256, 256))
    else:
        tiles_to_mosaic = tiles[tiles.intersects(box)]
        files_to_mosaic = tiles_to_mosaic.path.apply(
            lambda x: rasterio.open(x)).tolist()

        mosaic, _ = rasterio.merge.merge(files_to_mosaic)
        return mosaic[:, row:row + 256, col:col + 256]


def extract_subset_file(tiles, box, path):
    """Extract subset and store as georeferened jpeg."""
    if Path(path).exists():
        return
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
        dst.write(extract_subset(tiles, box))
