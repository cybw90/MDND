import resource
import tracemalloc

from functools import wraps


def track_memory(func):
    @wraps(func)
    def inner(*args, **kwargs):
        # starting the monitoring
        tracemalloc.start()
        result = func(*args, **kwargs)
        # displaying the memory
        current, peak = tracemalloc.get_traced_memory()

        print()
        print("Memory Usage for Function", func.__name__)
        print("Current", round(current / 1024, 4), "MB", end="\t")
        print("Peak", round(peak / 1024, 4), "MB")
        print("Current", round(current / 1024 / 1024, 4), "GB", end="\t")
        print("Peak", round(peak / 1024 / 1024, 4), "GB")
        print()

        tracemalloc.stop()
        return result

    return inner


def limit_memory(maxsize):
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (maxsize, hard))


def reset_limit_memory():
    resource.setrlimit(resource.RLIMIT_AS, (-1, -1))
