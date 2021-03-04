from gitdir.gitdir.gitdir import download as get_subdirectory
colmap_url = r"https://github.com/colmap/colmap/tree/78b6ae707e79542d34729e95052c10f6310c050b/scripts/python"
get_subdirectory(colmap_url, flatten=True, output_dir=r"./colmap_scripts")