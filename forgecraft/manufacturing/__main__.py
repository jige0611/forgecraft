"""允许 `python -m forgecraft.manufacturing` 直接调用导出 CLI。

用法:
  python -m forgecraft.manufacturing --body-file design_output/best_body.json --catalog speedster
"""

from forgecraft.manufacturing import main

if __name__ == "__main__":
    main()
