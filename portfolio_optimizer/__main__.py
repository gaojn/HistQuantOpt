"""支持 `python -m portfolio_optimizer ...`（等价于 hqopt，不依赖 PATH）。"""
from portfolio_optimizer.cli import main

if __name__ == "__main__":
    main()
