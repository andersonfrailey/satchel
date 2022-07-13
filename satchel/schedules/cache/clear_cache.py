from pathlib import Path


CUR_PATH = Path(__file__).resolve().parent


def clear_cache():
    """
    Deletes all previous versions of the cached schedules
    """
    for _file in CUR_PATH.glob("schedule*.csv"):
        _file.unlink()
