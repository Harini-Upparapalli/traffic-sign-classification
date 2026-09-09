"""
dataset_generator.py
Generates clean, synthetic traffic sign training images and preset test samples.
Covers 17 major traffic sign categories with data augmentation.
"""

import os
import math
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

# Define 17 comprehensive sign classes
SIGN_CLASSES = [
    "stop_sign",
    "yield_sign",
    "pedestrian_crossing",
    "school_zone",
    "kids_play_area",
    "bus_stop",
    "traffic_signal",
    "road_work",
    "turn_right",
    "turn_left",
    "no_u_turn",
    "u_turn",
    "speed_limit_50",
    "speed_limit_30",
    "no_entry",
    "no_parking",
    "roundabout"
]

CLASS_DISPLAY_NAMES = {
    "stop_sign": "Stop Sign",
    "yield_sign": "Yield / Give Way Sign",
    "pedestrian_crossing": "Pedestrian Crossing Warning",
    "school_zone": "School Zone / School Crossing",
    "kids_play_area": "Children / Playground Ahead",
    "bus_stop": "Bus Stop / Bus Zone",
    "traffic_signal": "Traffic Light Ahead",
    "road_work": "Road Work / Work in Progress",
    "turn_right": "Turn Right Ahead",
    "turn_left": "Turn Left Ahead",
    "no_u_turn": "No U-Turn Sign",
    "u_turn": "U-Turn Permitted / Mandatory",
    "speed_limit_50": "Speed Limit 50 km/h",
    "speed_limit_30": "Speed Limit 30 km/h",
    "no_entry": "No Entry Sign",
    "no_parking": "No Parking",
    "roundabout": "Roundabout Ahead"
}

CLASS_CATEGORIES = {
    "stop_sign": "REGULATORY / MANDATORY",
    "yield_sign": "REGULATORY / GIVE WAY",
    "pedestrian_crossing": "WARNING SIGN",
    "school_zone": "WARNING SIGN / SCHOOL",
    "kids_play_area": "WARNING SIGN / PLAYGROUND",
    "bus_stop": "INFORMATION / MANDATORY",
    "traffic_signal": "WARNING SIGN",
    "road_work": "WARNING SIGN / CONSTRUCTION",
    "turn_right": "MANDATORY SIGN",
    "turn_left": "MANDATORY SIGN",
    "no_u_turn": "PROHIBITORY SIGN",
    "u_turn": "MANDATORY / PERMISSIVE",
    "speed_limit_50": "REGULATORY / SPEED LIMIT",
    "speed_limit_30": "REGULATORY / SPEED LIMIT",
    "no_entry": "PROHIBITORY SIGN",
    "no_parking": "PROHIBITORY SIGN",
    "roundabout": "MANDATORY SIGN"
}

CLASS_DESCRIPTIONS = {
    "stop_sign": "Octagonal red sign requiring drivers to come to a complete stop at the stop line or intersection before proceeding safely.",
    "yield_sign": "Inverted triangular sign indicating drivers must yield the right-of-way to traffic on the main road.",
    "pedestrian_crossing": "Warning sign indicating pedestrians frequently cross the roadway ahead. Reduce speed and prepare to stop.",
    "school_zone": "Warning sign indicating a school zone or school crossing ahead. Slow down and watch for children crossing.",
    "kids_play_area": "Warning sign indicating a playground or children playing area ahead. Drive with extreme caution.",
    "bus_stop": "Information sign indicating a designated bus stop or bus zone. Do not block or park in this zone.",
    "traffic_signal": "Warning sign indicating automated traffic signals ahead. Prepare to stop if light is red.",
    "road_work": "Warning sign indicating road construction, maintenance, or workers ahead. Reduce speed and obey detour signals.",
    "turn_right": "Mandatory direction sign indicating traffic must turn right ahead.",
    "turn_left": "Mandatory direction sign indicating traffic must turn left ahead.",
    "no_u_turn": "Prohibitory sign strictly forbidding vehicles from executing a U-turn at this location.",
    "u_turn": "Mandatory/permissive sign indicating a designated U-turn lane or U-turn point ahead.",
    "speed_limit_50": "Regulatory maximum speed limit sign enforcing a maximum allowed speed of 50 km/h.",
    "speed_limit_30": "Regulatory maximum speed limit sign enforcing a maximum allowed speed of 30 km/h (common in urban/zone areas).",
    "no_entry": "Prohibitory sign indicating vehicles are strictly forbidden from entering the street or roadway.",
    "no_parking": "Prohibitory sign indicating parking is not allowed in this area. Drivers must not park vehicles where this sign is displayed.",
    "roundabout": "Mandatory sign indicating a circular intersection ahead. Traffic flows counter-clockwise."
}


def draw_regular_polygon(draw, center, radius, n_sides, rotation_deg=0, fill=None, outline=None, width=1):
    """Draw a regular polygon with n sides."""
    cx, cy = center
    points = []
    angle_offset = math.radians(rotation_deg)
    for i in range(n_sides):
        angle = angle_offset + (2 * math.pi * i / n_sides)
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        points.append((x, y))
    draw.polygon(points, fill=fill, outline=outline, width=width)
    return points


def create_base_traffic_sign(sign_type, img_size=(256, 256)):
    """Draw clean base vector-like traffic sign using PIL."""
    width, height = img_size
    image = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    cx, cy = width // 2, height // 2
    r = min(width, height) // 2 - 16

    try:
        font_large = ImageFont.truetype("arial.ttf", int(r * 0.55))
        font_bold = ImageFont.truetype("arialbd.ttf", int(r * 0.55))
    except Exception:
        font_large = ImageFont.load_default()
        font_bold = ImageFont.load_default()

    if sign_type == "stop_sign":
        draw_regular_polygon(draw, (cx, cy), r, n_sides=8, rotation_deg=22.5, fill=(217, 30, 24, 255))
        draw_regular_polygon(draw, (cx, cy), int(r * 0.94), n_sides=8, rotation_deg=22.5, fill=None, outline=(255, 255, 255, 255), width=int(r * 0.06))
        text = "STOP"
        bbox = draw.textbbox((0, 0), text, font=font_bold)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((cx - tw / 2, cy - th / 2 - r * 0.05), text, fill=(255, 255, 255, 255), font=font_bold)

    elif sign_type == "yield_sign":
        pts = [(cx - r * 1.05, cy - r * 0.75), (cx + r * 1.05, cy - r * 0.75), (cx, cy + r * 1.05)]
        draw.polygon(pts, fill=(217, 30, 24, 255))
        inner_pts = [(cx - r * 0.7, cy - r * 0.55), (cx + r * 0.7, cy - r * 0.55), (cx, cy + r * 0.7)]
        draw.polygon(inner_pts, fill=(255, 255, 255, 255))

    elif sign_type == "pedestrian_crossing":
        pts = [(cx, cy - r * 1.05), (cx - r * 1.05, cy + r * 0.85), (cx + r * 1.05, cy + r * 0.85)]
        draw.polygon(pts, fill=(217, 30, 24, 255))
        inner_pts = [(cx, cy - r * 0.65), (cx - r * 0.72, cy + r * 0.65), (cx + r * 0.72, cy + r * 0.65)]
        draw.polygon(inner_pts, fill=(255, 255, 255, 255))

        line_y = cy + r * 0.45
        for x_off in [-r*0.4, -r*0.15, r*0.1, r*0.35]:
            draw.rectangle([cx + x_off, line_y, cx + x_off + r*0.18, line_y + r*0.1], fill=(20, 20, 20, 255))

        head_r = int(r * 0.12)
        draw.ellipse([cx - head_r, cy - r * 0.38 - head_r, cx + head_r, cy - r * 0.38 + head_r], fill=(20, 20, 20, 255))
        draw.line([(cx - r * 0.05, cy - r * 0.22), (cx + r * 0.08, cy + r * 0.1)], fill=(20, 20, 20, 255), width=int(r * 0.12))
        draw.line([(cx - r * 0.22, cy - r * 0.1), (cx + r * 0.22, cy - r * 0.18)], fill=(20, 20, 20, 255), width=int(r * 0.08))
        draw.line([(cx + r * 0.08, cy + r * 0.1), (cx - r * 0.25, cy + r * 0.42)], fill=(20, 20, 20, 255), width=int(r * 0.1))
        draw.line([(cx + r * 0.08, cy + r * 0.1), (cx + r * 0.32, cy + r * 0.42)], fill=(20, 20, 20, 255), width=int(r * 0.1))

    elif sign_type == "school_zone":
        # Yellow warning pentagon or red/yellow triangle with 2 student figures
        pts = [(cx, cy - r * 1.05), (cx - r * 1.05, cy + r * 0.85), (cx + r * 1.05, cy + r * 0.85)]
        draw.polygon(pts, fill=(217, 30, 24, 255))
        inner_pts = [(cx, cy - r * 0.65), (cx - r * 0.72, cy + r * 0.65), (cx + r * 0.72, cy + r * 0.65)]
        draw.polygon(inner_pts, fill=(250, 210, 40, 255)) # Yellow interior

        # Figure 1 (Tall Student with bag)
        cx1 = cx - r * 0.18
        draw.ellipse([cx1 - r*0.08, cy - r*0.35 - r*0.08, cx1 + r*0.08, cy - r*0.35 + r*0.08], fill=(20, 20, 20, 255))
        draw.line([(cx1, cy - r*0.25), (cx1, cy + r*0.18)], fill=(20, 20, 20, 255), width=int(r * 0.1))
        draw.line([(cx1, cy + r*0.18), (cx1 - r*0.12, cy + r*0.45)], fill=(20, 20, 20, 255), width=int(r * 0.08))
        draw.line([(cx1, cy + r*0.18), (cx1 + r*0.1, cy + r*0.45)], fill=(20, 20, 20, 255), width=int(r * 0.08))
        draw.rectangle([cx1 - r*0.18, cy - r*0.1, cx1 - r*0.08, cy + r*0.1], fill=(20, 20, 20, 255)) # backpack

        # Figure 2 (Smaller Student)
        cx2 = cx + r * 0.22
        draw.ellipse([cx2 - r*0.07, cy - r*0.25 - r*0.07, cx2 + r*0.07, cy - r*0.25 + r*0.07], fill=(20, 20, 20, 255))
        draw.line([(cx2, cy - r*0.18), (cx2, cy + r*0.18)], fill=(20, 20, 20, 255), width=int(r * 0.08))
        draw.line([(cx2, cy + r*0.18), (cx2 - r*0.08, cy + r*0.45)], fill=(20, 20, 20, 255), width=int(r * 0.07))
        draw.line([(cx2, cy + r*0.18), (cx2 + r*0.12, cy + r*0.45)], fill=(20, 20, 20, 255), width=int(r * 0.07))

    elif sign_type == "kids_play_area":
        # Yellow/Red Warning Triangle with child chasing a ball
        pts = [(cx, cy - r * 1.05), (cx - r * 1.05, cy + r * 0.85), (cx + r * 1.05, cy + r * 0.85)]
        draw.polygon(pts, fill=(217, 30, 24, 255))
        inner_pts = [(cx, cy - r * 0.65), (cx - r * 0.72, cy + r * 0.65), (cx + r * 0.72, cy + r * 0.65)]
        draw.polygon(inner_pts, fill=(250, 210, 40, 255))

        # Child running
        draw.ellipse([cx - r*0.1, cy - r*0.35 - r*0.1, cx + r*0.1, cy - r*0.35 + r*0.1], fill=(20, 20, 20, 255))
        draw.line([(cx, cy - r*0.22), (cx - r*0.1, cy + r*0.15)], fill=(20, 20, 20, 255), width=int(r * 0.1))
        # Running legs
        draw.line([(cx - r*0.1, cy + r*0.15), (cx - r*0.3, cy + r*0.4)], fill=(20, 20, 20, 255), width=int(r * 0.09))
        draw.line([(cx - r*0.1, cy + r*0.15), (cx + r*0.18, cy + r*0.4)], fill=(20, 20, 20, 255), width=int(r * 0.09))
        # Ball circle
        draw.ellipse([cx + r*0.25, cy + r*0.2, cx + r*0.42, cy + r*0.37], fill=(20, 20, 20, 255))

    elif sign_type == "bus_stop":
        # Solid Blue Square / Rectangle sign with white bus icon
        draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=(13, 71, 161, 255))
        draw.rectangle([cx - int(r*0.9), cy - int(r*0.9), cx + int(r*0.9), cy + int(r*0.9)], outline=(255, 255, 255, 255), width=int(r*0.06))
        # White Bus Body
        bw, bh = int(r * 1.1), int(r * 0.75)
        draw.rectangle([cx - bw//2, cy - bh//2, cx + bw//2, cy + bh//2], fill=(255, 255, 255, 255))
        # Bus Windshield (blue glass)
        draw.rectangle([cx - int(bw*0.4), cy - int(bh*0.35), cx + int(bw*0.4), cy - int(bh*0.05)], fill=(13, 71, 161, 255))
        # Headlights
        draw.ellipse([cx - int(bw*0.4), cy + int(bh*0.15), cx - int(bw*0.25), cy + int(bh*0.35)], fill=(245, 180, 0, 255))
        draw.ellipse([cx + int(bw*0.25), cy + int(bh*0.15), cx + int(bw*0.4), cy + int(bh*0.35)], fill=(245, 180, 0, 255))
        # Wheels
        wr = int(r * 0.12)
        draw.ellipse([cx - int(bw*0.35), cy + bh//2 - wr//2, cx - int(bw*0.15), cy + bh//2 + wr], fill=(20, 20, 20, 255))
        draw.ellipse([cx + int(bw*0.15), cy + bh//2 - wr//2, cx + int(bw*0.35), cy + bh//2 + wr], fill=(20, 20, 20, 255))

    elif sign_type == "road_work":
        # Red/Yellow Warning Triangle with Worker Shoveling
        pts = [(cx, cy - r * 1.05), (cx - r * 1.05, cy + r * 0.85), (cx + r * 1.05, cy + r * 0.85)]
        draw.polygon(pts, fill=(217, 30, 24, 255))
        inner_pts = [(cx, cy - r * 0.65), (cx - r * 0.72, cy + r * 0.65), (cx + r * 0.72, cy + r * 0.65)]
        draw.polygon(inner_pts, fill=(255, 255, 255, 255))

        # Worker head & body bent forward
        draw.ellipse([cx - r*0.15 - r*0.09, cy - r*0.2 - r*0.09, cx - r*0.15 + r*0.09, cy - r*0.2 + r*0.09], fill=(20, 20, 20, 255))
        draw.line([(cx - r*0.15, cy - r*0.1), (cx + r*0.1, cy + r*0.1)], fill=(20, 20, 20, 255), width=int(r * 0.11))
        # Shovel stick & shovel head pile
        draw.line([(cx - r*0.1, cy + r*0.05), (cx + r*0.35, cy + r*0.35)], fill=(20, 20, 20, 255), width=int(r * 0.08))
        draw.polygon([(cx + r*0.3, cy + r*0.3), (cx + r*0.45, cy + r*0.35), (cx + r*0.35, cy + r*0.48)], fill=(20, 20, 20, 255))

    elif sign_type == "turn_left":
        # Solid Blue Circle with white arrow pointing left
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(13, 71, 161, 255))
        draw.rectangle([cx - r * 0.2, cy - r * 0.14, cx + r * 0.45, cy + r * 0.14], fill=(255, 255, 255, 255))
        arrow_head = [
            (cx - r * 0.15, cy - r * 0.38),
            (cx - r * 0.55, cy),
            (cx - r * 0.15, cy + r * 0.38)
        ]
        draw.polygon(arrow_head, fill=(255, 255, 255, 255))

    elif sign_type == "turn_right":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(13, 71, 161, 255))
        draw.rectangle([cx - r * 0.45, cy - r * 0.14, cx + r * 0.2, cy + r * 0.14], fill=(255, 255, 255, 255))
        arrow_head = [
            (cx + r * 0.15, cy - r * 0.38),
            (cx + r * 0.55, cy),
            (cx + r * 0.15, cy + r * 0.38)
        ]
        draw.polygon(arrow_head, fill=(255, 255, 255, 255))

    elif sign_type == "no_u_turn":
        # Red ring circle, white center, black U-turn arrow, red diagonal slash
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(217, 30, 24, 255))
        draw.ellipse([cx - int(r*0.75), cy - int(r*0.75), cx + int(r*0.75), cy + int(r*0.75)], fill=(255, 255, 255, 255))
        # Draw U-turn curve
        draw.arc([cx - r*0.35, cy - r*0.4, cx + r*0.25, cy + r*0.2], start=180, end=0, fill=(20, 20, 20, 255), width=int(r*0.14))
        draw.line([(cx - r*0.35, cy - r*0.1), (cx - r*0.35, cy + r*0.35)], fill=(20, 20, 20, 255), width=int(r*0.14))
        draw.line([(cx + r*0.25, cy - r*0.1), (cx + r*0.25, cy + r*0.35)], fill=(20, 20, 20, 255), width=int(r*0.14))
        # U-turn arrow head pointing down on left leg
        draw.polygon([(cx - r*0.5, cy + r*0.2), (cx - r*0.2, cy + r*0.2), (cx - r*0.35, cy + r*0.48)], fill=(20, 20, 20, 255))
        # Red diagonal slash
        draw.line([(cx - r * 0.65, cy - r * 0.65), (cx + r * 0.65, cy + r * 0.65)], fill=(217, 30, 24, 255), width=int(r * 0.16))

    elif sign_type == "u_turn":
        # Solid Blue Circle with white U-turn arrow
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(13, 71, 161, 255))
        draw.arc([cx - r*0.35, cy - r*0.4, cx + r*0.25, cy + r*0.2], start=180, end=0, fill=(255, 255, 255, 255), width=int(r*0.16))
        draw.line([(cx - r*0.35, cy - r*0.1), (cx - r*0.35, cy + r*0.35)], fill=(255, 255, 255, 255), width=int(r*0.16))
        draw.line([(cx + r*0.25, cy - r*0.1), (cx + r*0.25, cy + r*0.35)], fill=(255, 255, 255, 255), width=int(r*0.16))
        draw.polygon([(cx - r*0.52, cy + r*0.2), (cx - r*0.18, cy + r*0.2), (cx - r*0.35, cy + r*0.48)], fill=(255, 255, 255, 255))

    elif sign_type == "speed_limit_50":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(217, 30, 24, 255))
        draw.ellipse([cx - int(r*0.75), cy - int(r*0.75), cx + int(r*0.75), cy + int(r*0.75)], fill=(255, 255, 255, 255))
        text = "50"
        bbox = draw.textbbox((0, 0), text, font=font_bold)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((cx - tw / 2, cy - th / 2 - r * 0.05), text, fill=(20, 20, 20, 255), font=font_bold)

    elif sign_type == "speed_limit_30":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(217, 30, 24, 255))
        draw.ellipse([cx - int(r*0.75), cy - int(r*0.75), cx + int(r*0.75), cy + int(r*0.75)], fill=(255, 255, 255, 255))
        text = "30"
        bbox = draw.textbbox((0, 0), text, font=font_bold)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((cx - tw / 2, cy - th / 2 - r * 0.05), text, fill=(20, 20, 20, 255), font=font_bold)

    elif sign_type == "no_parking":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(13, 71, 161, 255))
        text = "P"
        bbox = draw.textbbox((0, 0), text, font=font_bold)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((cx - tw / 2, cy - th / 2 - r * 0.05), text, fill=(255, 255, 255, 255), font=font_bold)
        slash_w = int(r * 0.18)
        draw.line([(cx - r * 0.75, cy - r * 0.75), (cx + r * 0.75, cy + r * 0.75)], fill=(217, 30, 24, 255), width=slash_w)

    elif sign_type == "no_entry":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(217, 30, 24, 255))
        bar_w = int(r * 1.5)
        bar_h = int(r * 0.38)
        draw.rectangle([cx - bar_w // 2, cy - bar_h // 2, cx + bar_w // 2, cy + bar_h // 2], fill=(255, 255, 255, 255))

    elif sign_type == "roundabout":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(13, 71, 161, 255))
        for angle_deg in [0, 120, 240]:
            rad = math.radians(angle_deg)
            acx = cx + r * 0.45 * math.cos(rad)
            acy = cy + r * 0.45 * math.sin(rad)
            draw.ellipse([acx - r * 0.15, acy - r * 0.15, acx + r * 0.15, acy + r * 0.15], fill=(255, 255, 255, 255))

    elif sign_type == "traffic_signal":
        pts = [(cx, cy - r * 1.05), (cx - r * 1.05, cy + r * 0.85), (cx + r * 1.05, cy + r * 0.85)]
        draw.polygon(pts, fill=(217, 30, 24, 255))
        inner_pts = [(cx, cy - r * 0.65), (cx - r * 0.72, cy + r * 0.65), (cx + r * 0.72, cy + r * 0.65)]
        draw.polygon(inner_pts, fill=(255, 255, 255, 255))

        box_w, box_h = int(r * 0.28), int(r * 0.75)
        draw.rectangle([cx - box_w // 2, cy - box_h // 2 + int(r * 0.05), cx + box_w // 2, cy + box_h // 2 + int(r * 0.05)], fill=(30, 30, 30, 255))
        dot_r = int(r * 0.08)
        draw.ellipse([cx - dot_r, cy - r * 0.2 - dot_r, cx + dot_r, cy - r * 0.2 + dot_r], fill=(235, 40, 40, 255))
        draw.ellipse([cx - dot_r, cy + r * 0.05 - dot_r, cx + dot_r, cy + r * 0.05 + dot_r], fill=(245, 180, 0, 255))
        draw.ellipse([cx - dot_r, cy + r * 0.3 - dot_r, cx + dot_r, cy + r * 0.3 + dot_r], fill=(40, 190, 70, 255))

    background = Image.new("RGB", img_size, (245, 247, 250))
    background.paste(image, mask=image.split()[3])
    return background


def augment_image(image):
    """Apply realistic noise, rotations, scale variations, and margin padding shifts for dataset augmentation."""
    img = image.copy()
    w, h = img.size

    angle = random.uniform(-20, 20)
    bg_color = (random.randint(235, 255), random.randint(235, 255), random.randint(235, 255))
    img = img.rotate(angle, resample=Image.BICUBIC, fillcolor=bg_color)

    scale = random.uniform(0.7, 1.3)
    new_w, new_h = max(10, int(w * scale)), max(10, int(h * scale))
    resized = img.resize((new_w, new_h), Image.BICUBIC)

    canvas = Image.new("RGB", (w, h), bg_color)
    offset_x = (w - new_w) // 2 + random.randint(-12, 12)
    offset_y = (h - new_h) // 2 + random.randint(-12, 12)
    canvas.paste(resized, (offset_x, offset_y))
    img = canvas

    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.75, 1.25))

    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(random.uniform(0.75, 1.25))

    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(random.uniform(0.8, 1.2))

    if random.random() > 0.4:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.0)))

    return img


def generate_dataset(output_dir="dataset", samples_per_class=70, save_preset_samples=True):
    """Generate training dataset directory with augmented traffic sign images."""
    os.makedirs(output_dir, exist_ok=True)

    sample_dir = os.path.join(os.path.dirname(output_dir), "samples")
    if save_preset_samples:
        os.makedirs(sample_dir, exist_ok=True)

    print(f"Generating synthetic traffic sign dataset across {len(SIGN_CLASSES)} classes in '{output_dir}'...")

    for sign_type in SIGN_CLASSES:
        class_dir = os.path.join(output_dir, sign_type)
        os.makedirs(class_dir, exist_ok=True)

        base_img = create_base_traffic_sign(sign_type, img_size=(200, 200))

        if save_preset_samples:
            sample_path = os.path.join(sample_dir, f"{sign_type}.png")
            base_img.save(sample_path)

        base_img.save(os.path.join(class_dir, f"{sign_type}_00.png"))

        for i in range(1, samples_per_class):
            aug_img = augment_image(base_img)
            aug_img.save(os.path.join(class_dir, f"{sign_type}_{i:02d}.png"))

    print(f"Successfully generated {len(SIGN_CLASSES) * samples_per_class} images across {len(SIGN_CLASSES)} classes!")


if __name__ == "__main__":
    generate_dataset()
