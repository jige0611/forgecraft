"""Add `import logging` + `_logger = logging.getLogger(__name__)` to files missing it."""
import os, re

# Key modules that SHOULD have logging (skip test/init/meta files)
ADD_LOGGER = [
    "forgecraft/core/flexible_components.py",
    "forgecraft/core/materials_database.py",
    "forgecraft/core/motor_adapter.py",
    "forgecraft/core/joint_optimizer.py",
    "forgecraft/core/loader.py",
    "forgecraft/core/generator.py",
    "forgecraft/core/morphology.py",
    "forgecraft/core/catalog.py",
    "forgecraft/core/engineering_dc_motor.py",
    "forgecraft/evaluation/fitness.py",
    "forgecraft/evaluation/gpu_engine.py",
    "forgecraft/evaluation/metrics.py",
    "forgecraft/simulation/terrain_curriculum.py",
    "forgecraft/evolution/operators.py",
    "forgecraft/evolution/selection.py",
    "forgecraft/evolution/adaptive_mutation.py",
    "forgecraft/evolution/map_elites.py",
    "forgecraft/experiment/management.py",
    "forgecraft/config.py",
    "forgecraft/interactive.py",
    "forgecraft/dashboard.py",
    "forgecraft/dashboard_live.py",
    "forgecraft/main.py",
]

for rel in ADD_LOGGER:
    fpath = os.path.join(os.getcwd(), rel)
    if not os.path.exists(fpath):
        print(f"NOT FOUND: {rel}")
        continue
    
    with open(fpath, 'r', encoding='utf-8') as f:
        content = f.read()
        lines = content.split('\n')
    
    # Check if already has logger
    if '_logger = logging.getLogger' in content or 'logger = logging.getLogger' in content:
        print(f"SKIP (has logger): {rel}")
        continue
    
    # Check if already imports logging
    has_import = 'import logging' in content
    
    # Insert import logging if needed
    new_lines = []
    import_added = False
    logger_added = False
    logger_name = '_logger'
    
    for i, line in enumerate(lines):
        new_lines.append(line)
        
        # After the last import statement, add import logging
        stripped = line.strip()
        if not import_added and not has_import:
            # Find the last import line
            is_import = stripped.startswith("import ") or stripped.startswith("from ")
            is_continue = stripped.startswith("(") or not stripped  # continuation of multi-line import
            next_is_blank = i + 1 < len(lines) and not lines[i + 1].strip()
            next_is_code = i + 1 < len(lines) and (
                lines[i + 1].strip().startswith("#") or
                lines[i + 1].strip().startswith("class ") or
                lines[i + 1].strip().startswith("def ") or
                lines[i + 1].strip().startswith("@") or
                lines[i + 1].strip().startswith('"""') or
                lines[i + 1].strip().startswith("__all__") or
                (lines[i + 1].strip() and not lines[i + 1].strip().startswith("from ") and not lines[i + 1].strip().startswith("import "))
            )
            if is_import and not is_continue and next_is_code:
                new_lines.append("import logging")
                import_added = True
    
    # Add logger line after imports and optionally after __all__
    lines = new_lines
    new_lines = []
    for i, line in enumerate(lines):
        new_lines.append(line)
        stripped = line.strip()
        
        if not logger_added:
            # Find a good insertion point: after __all__ block or after last import
            if stripped.startswith("]") and i > 0:
                # Check if previous lines look like __all__
                prev_content = '\n'.join(lines[max(0,i-15):i])
                if '__all__' in prev_content:
                    new_lines.append("")
                    new_lines.append(f"{logger_name} = logging.getLogger(__name__)")
                    new_lines.append("")
                    logger_added = True
                    continue
            
            # If no __all__, after last import before first class/def/#
            is_import = stripped.startswith("import ") or stripped.startswith("from ")
            if is_import:
                next_non_blank = ""
                for j in range(i+1, min(i+5, len(lines))):
                    if lines[j].strip():
                        next_non_blank = lines[j].strip()
                        break
                if next_non_blank and not next_non_blank.startswith("from ") and not next_non_blank.startswith("import ") and not next_non_blank.startswith("("):
                    if not any("logging.getLogger" in l for l in lines[max(0,i-2):i+2]):
                        new_lines.append("")
                        new_lines.append(f"{logger_name} = logging.getLogger(__name__)")
                        new_lines.append("")
                        logger_added = True
    
    if not logger_added:
        print(f"WARN: could not find insertion point in {rel}")
        continue
    
    with open(fpath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(new_lines))
    
    print(f"OK: {rel}" + (" (import added)" if import_added else ""))

print("\nDone!")
