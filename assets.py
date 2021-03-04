from pathlib import Path
import requests
from tqdm import tqdm
import click
import os

STAC = "https://data.geo.admin.ch/api/stac/v0.9/"


@click.command()
@click.argument("collection")
@click.option("--outdir", "-o")
def download_assets(collection, outdir=os.getcwd()):
    """Download (tiled) assets of a collection through the STAC api.

    Example:
        $ python3 assets.py ch.swisstopo.swissimage-dop10 -o /data/swissimage
    """
    url = STAC + "collections/{}/items".format(collection)
    features = []
    while url:
        r = requests.get(url).json()
        features.extend(r["features"])
        url = None
        for link in r["links"]:
            if link["rel"] == "next":
                url = link["href"]

    for item in tqdm(features):
        for asset in item["assets"]:
            if item["collection"] == "ch.swisstopo.swissimage-dop10":
                # only get the 10 cm resolution assets
                if asset != item["id"] + "_0.1_2056.tif":
                    continue
            else:
                url = item["assets"][asset]["href"]
                outpath = outdir.joinpath(Path(url).name)
                if outpath.exists():
                    continue
                else:
                    r = requests.get(url)
                    r.raise_for_status()
                    outpath.write_bytes(r.content)


if __name__ == "__main__":
    # download_assets('ch.swisstopo.swissimage-dop10', '/data/swissimage')
    download_assets()
