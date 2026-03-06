from dtst.engines.base import SearchEngine
from dtst.engines.brave import BraveSearchEngine
from dtst.engines.flickr import FlickrEngine
from dtst.engines.serper import SerperEngine
from dtst.engines.wikimedia import WikimediaEngine

ENGINE_REGISTRY: dict[str, type[SearchEngine]] = {
    "brave": BraveSearchEngine,
    "flickr": FlickrEngine,
    "serper": SerperEngine,
    "wikimedia": WikimediaEngine,
}
