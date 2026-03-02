from dtst.engines.base import SearchEngine
from dtst.engines.bing import BingEngine
from dtst.engines.flickr import FlickrEngine
from dtst.engines.serper import SerperEngine
from dtst.engines.wikimedia import WikimediaEngine

ENGINE_REGISTRY: dict[str, type[SearchEngine]] = {
    "flickr": FlickrEngine,
    "serper": SerperEngine,
    "bing": BingEngine,
    "wikimedia": WikimediaEngine,
}
