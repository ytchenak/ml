import os
from PIL import Image, ImageDraw, ImageFont

# Directories
orig_dir = 'test_images/urban100'
x2_dir = 'test_images/urban100x2'
x4_dir = 'test_images/urban100x4'
sr2x_dir = 'results/swin2sr_classical_sr_x2'
ir2x_dir = 'results/swinir_classical_sr_x2'
sr4x_dir = 'results/swin2sr_real_sr_x4'
ir4x_dir = 'results/swinir_real_sr_x4'
output_dir = 'concatenated_results'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Labels for each column (order must match image order)
labels = [
    'Original',
    'LR x2 (Qubic Interpolation)',
    'Swin2SR classical x2',
    'SwinIR classical x2',
    'LR x4 (Qubic Interpolation)',
    'Swin2SR real-world x4',
    'SwinIR real-world x4',
]

# Try to load a default font
try:
    font = ImageFont.truetype("arial.ttf", 32)
except:
    font = ImageFont.load_default()

def pad_to_same_height(images):
    max_height = max(img.height for img in images)
    padded = []
    for img in images:
        if img.height < max_height:
            new_img = Image.new('RGB', (img.width, max_height), (0, 0, 0))
            new_img.paste(img, (0, (max_height - img.height) // 2))
            padded.append(new_img)
        else:
            padded.append(img)
    return padded

def add_labels_to_row(images, labels, font):
    # Add a label above each image
    label_height = 40
    total_width = sum(img.width for img in images)
    max_height = max(img.height for img in images)
    labeled_img = Image.new('RGB', (total_width, max_height + label_height), (255, 255, 255))
    draw = ImageDraw.Draw(labeled_img)
    x_offset = 0
    for img, label in zip(images, labels):
        # Center label above image
        w = img.width
        # Robust text size calculation
        try:
            bbox = draw.textbbox((0, 0), label, font=font)
            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            text_w, text_h = draw.textsize(label, font=font)
        text_x = x_offset + (w - text_w) // 2
        text_y = (label_height - text_h) // 2
        draw.text((text_x, text_y), label, fill=(0, 0, 0), font=font)
        labeled_img.paste(img, (x_offset, label_height))
        x_offset += w
    return labeled_img

def concat_row(images):
    # Concatenate images horizontally without labels
    total_width = sum(img.width for img in images)
    max_height = max(img.height for img in images)
    concat_img = Image.new('RGB', (total_width, max_height), (255, 255, 255))
    x_offset = 0
    for img in images:
        concat_img.paste(img, (x_offset, 0))
        x_offset += img.width
    return concat_img

# Store all concatenated rows
all_rows = []
first_row_widths = None

for i in range(1, 101):
    idx = f'{i:03d}'
    orig_path = os.path.join(orig_dir, f'img_{idx}.png')
    x2_path = os.path.join(x2_dir, f'img_{idx}x2.png')
    x4_path = os.path.join(x4_dir, f'img_{idx}.png')
    sr2x_path = os.path.join(sr2x_dir, f'img_{idx}_Swin2SR.png')
    ir2x_path = os.path.join(ir2x_dir, f'img_{idx}_SwinIR.png')
    sr4x_path = os.path.join(sr4x_dir, f'img_{idx}_Swin2SR.png')
    ir4x_path = os.path.join(ir4x_dir, f'img_{idx}_SwinIR.png')

    # Load images in the new order: orig, sr2x, x2, sr4x, x4
    imgs = []
    for path in [orig_path, x2_path, sr2x_path, ir2x_path, x4_path, sr4x_path, ir4x_path]:
        if not os.path.exists(path):
            print(f'Missing: {path}')
            imgs.append(Image.new('RGB', (256, 256), (255, 0, 0)))  # Red placeholder
        else:
            imgs.append(Image.open(path).convert('RGB'))

    # Upscale x2 and x4 reduced images to original size
    orig_size = imgs[0].size
    imgs[1] = imgs[1].resize(orig_size, Image.LANCZOS)  # x2 reduced
    imgs[4] = imgs[4].resize(orig_size, Image.LANCZOS)  # x4 reduced

    # Pad to same height (shouldn't be needed now, but kept for safety)
    imgs = pad_to_same_height(imgs)

    # For the first row, record the width of each image
    if i == 1:
        first_row_widths = [img.width for img in imgs]
        row_img = add_labels_to_row(imgs, labels, font)
    else:
        # Resize each image to match the width of the corresponding image in the first row
        resized_imgs = [img.resize((first_row_widths[j], img.height), Image.LANCZOS) for j, img in enumerate(imgs)]
        row_img = concat_row(resized_imgs)
    all_rows.append(row_img)

    # Save individual row
    out_path = os.path.join(output_dir, f'concatenated_img_{idx}.png')
    row_img.save(out_path)
    print(f'Saved: {out_path}')

# Concatenate all rows vertically
if all_rows:
    total_height = sum(row.height for row in all_rows)
    max_width = max(row.width for row in all_rows)
    final_img = Image.new('RGB', (max_width, total_height), (255, 255, 255))
    y_offset = 0
    for row in all_rows:
        final_img.paste(row, (0, y_offset))
        y_offset += row.height
    final_img.save(os.path.join(output_dir, 'all_concatenated_vertical.png'))
    print(f'Saved vertical concatenation: {os.path.join(output_dir, 'all_concatenated_vertical.png')}')
