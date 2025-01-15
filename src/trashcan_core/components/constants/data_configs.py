IMSHAPE = (256, 480, 3)

INPUT_DIMS = (256, 256)

HUES = {
    "rov": 30,  # Distinct for robotic vehicle
    "plant": 120,  # Green for plants
    "animal_fish": 180,  # Blue for fish
    "animal_starfish": 300,  # Magenta for starfish
    "animal_shells": 240,  # Cyan for shells
    "animal_crab": 330,  # Pink-red for crabs
    "animal_eel": 210,  # Slightly different blue for eels
    "animal_etc": 270,  # Purple for other animals
    "trash_etc": 0,  # Red for other trash
    "trash_fabric": 45,  # Yellow for fabric
    "trash_fishing_gear": 90,  # Green for fishing gear
    "trash_metal": 15,  # Orange for metal
    "trash_paper": 60,  # Yellow-green for paper
    "trash_plastic": 135,  # Aqua for plastic
    "trash_rubber": 75,  # Olive for rubber
    "trash_wood": 150,  # Light green for wood
}

CATEGORIES = sorted(HUES.keys())

N_CLASSES = len(CATEGORIES)

assert IMSHAPE[0] % 32 == 0 and IMSHAPE[1] % 32 == 0, (
    "imshape should be multiples of 32. comment out to test different imshapes."
)
