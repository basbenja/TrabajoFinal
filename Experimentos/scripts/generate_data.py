import json
import os
import pprint

from data import DataGenerator
from constants import DATA_PARAMS_PATH, DATA_DIR

if __name__ == "__main__":
    # 1. Verify data directory exists. If not, create it.
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    else:
        print(f"The directory {DATA_DIR} already exists")

    # 2. Read the params for generating the new data
    with open(DATA_PARAMS_PATH, 'r') as f:
        params = json.load(f)
    pprint(params)

    # 3. Instatiate a DataGenerator. This creates the base folder where the
    # data is going to be stored for this group
    generator = DataGenerator(params)

    # 4. Generate the data
    data = generator.generate()

    # 5. Store the data
