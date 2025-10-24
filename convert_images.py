# Convert images from any extr
import os
from PIL import Image

# ----- CONFIG -----
root_dir = "train"                     # change if needed
folders_to_convert = ["NORMAL", "PNEUMONIA", "TUBERCULOSIS"]
# treat these as source formats (will be converted)
source_exts = ('.png', '.jpeg', '.bmp', '.tiff', '.jfif', '.webp', '.gif', '.jpe')
quality = 95                             # JPEG save quality
overwrite = False                        # True -> replace existing .jpg; False -> skip if .jpg exists
keep_original = False                    # True -> keep original file (don't delete); False -> remove it after conversion
# ------------------

for folder in folders_to_convert:
    folder_path = os.path.join(root_dir, folder)
    if not os.path.isdir(folder_path):
        print(f"Folder not found: {folder_path}")
        continue

    for filename in os.listdir(folder_path):
        src_path = os.path.join(folder_path, filename)
        if not os.path.isfile(src_path):
            continue

        name, ext = os.path.splitext(filename)
        ext_lower = ext.lower()

        # skip files already .jpg
        if ext_lower == ".jpg":
            continue

        # only process known source extensions
        if ext_lower not in source_exts:
            # optionally print or skip unknown filetypes
            print(f"Skipping (unknown ext): {filename}")
            continue

        new_name = name + ".jpg"
        dst_path = os.path.join(folder_path, new_name)

        # if target exists and we shouldn't overwrite -> skip
        if os.path.exists(dst_path) and not overwrite:
            print(f"Skipping {filename} -> {new_name} (target exists). Set overwrite=True to replace.")
            continue

        try:
            # open image and convert
            with Image.open(src_path) as img:
                # If the file is already JPEG in content (format == 'JPEG') and the extension is .jpeg/.jfif,
                # we can safely rename/move the file instead of re-encoding (faster, preserves metadata).
                if img.format == "JPEG" and ext_lower in ('.jpeg', '.jfif', '.jpe'):
                    # close then move/replace
                    img.close()
                    if os.path.exists(dst_path):
                        if overwrite:
                            os.replace(src_path, dst_path)   # replace existing
                        else:
                            print(f"Skipping rename for {filename} because {new_name} exists.")
                            continue
                    else:
                        os.rename(src_path, dst_path)
                    print(f"Renamed: {filename} -> {new_name} (was JPEG already)")
                    continue

                # Otherwise re-save as JPEG after converting to RGB
                rgb = img.convert("RGB")
                rgb.save(dst_path, "JPEG", quality=quality)

            # remove original if requested and if we didn't just rename it
            if not keep_original:
                try:
                    os.remove(src_path)
                except OSError:
                    print(f"Warning: couldn't remove original {src_path}")

            print(f"Converted: {filename} -> {new_name}")

        except Exception as e:
            print(f"Failed to convert {filename}: {e}")