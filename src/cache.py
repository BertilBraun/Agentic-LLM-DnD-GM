import hashlib
import os
import pickle
from functools import wraps


def generate_hash_code(data) -> str:
    # Serialize data with pickle
    serialized_data = pickle.dumps(data)

    # Create a hash object with MD5
    hash_object = hashlib.md5()
    hash_object.update(serialized_data)  # pickle.dumps returns bytes

    # Return the hash as a hex string
    return hash_object.hexdigest()


def cache_to_file(folder_name: str):
    # This decorator should be usable like @cache to cache the result of a function. The cache mapping should be stored in a file with the given file_name. The cache should be loaded at the beginning of the function and saved at the end of the function. The cache should be a dictionary that maps the arguments to the result of the function.
    folder_name = 'cache/' + folder_name

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            os.makedirs(folder_name, exist_ok=True)

            # Sort kwargs to ensure the hash is deterministic
            hash_code = generate_hash_code((func.__name__, args, sorted(kwargs.items())))
            cache_file_name = os.path.join(folder_name, hash_code + '.pkl')

            if os.path.exists(cache_file_name):
                with open(cache_file_name, 'rb') as f:
                    return pickle.load(f)

            result = func(*args, **kwargs)

            with open(cache_file_name, 'wb') as f:
                pickle.dump(result, f)

            return result

        return wrapper

    return decorator
