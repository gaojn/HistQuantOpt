"""支持 `python -m hqopt ...`（等价于 hqopt，不依赖 PATH）。"""
from hqopt.cli import main

if __name__ == "__main__":
    main()
