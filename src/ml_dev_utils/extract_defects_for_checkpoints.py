from pathlib import Path
from ml_dev_utils.common_utils.file_utils import get_files_in_dir



if __name__ == '__main__':

    filepath = Path("/home/alex/qi3/mir_rest_api/mir_api/api/v1/db_migration/data/checkpoint_json_files")

    files = get_files_in_dir(filepath, file_ext=".json")

    print(f"Number of files found {len(files)}")
    print(files)

    defect_to_checkpoint_map = {

        "GROUNDING_CIRCUIT": "decay at the lugs or terminals",
    }