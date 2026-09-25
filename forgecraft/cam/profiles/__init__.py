"""
打印机配置文件预设
"""

PRINTER_PROFILES = {
    "ender3_v2": {
        "plate_width": 220,
        "plate_depth": 220,
        "plate_height": 250,
        "nozzle_diameter": 0.4,
        "filament_diameter": 1.75,
        "max_speed": 150,
    },
    "prusa_mk3s": {
        "plate_width": 210,
        "plate_depth": 210,
        "plate_height": 210,
        "nozzle_diameter": 0.4,
        "filament_diameter": 1.75,
        "max_speed": 200,
    },
    "bambu_x1c": {
        "plate_width": 256,
        "plate_depth": 256,
        "plate_height": 256,
        "nozzle_diameter": 0.4,
        "filament_diameter": 1.75,
        "max_speed": 500,
        "extruder_count": 1,
    },
    "bambu_x1c_ams": {
        "plate_width": 256,
        "plate_depth": 256,
        "plate_height": 256,
        "nozzle_diameter": 0.4,
        "filament_diameter": 1.75,
        "max_speed": 500,
        "extruder_count": 4,
    },
    "voron_2.4": {
        "plate_width": 300,
        "plate_depth": 300,
        "plate_height": 300,
        "nozzle_diameter": 0.4,
        "filament_diameter": 1.75,
        "max_speed": 300,
    },
}
