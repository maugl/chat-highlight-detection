import json
import math


def main():
    log_out = {
        "progress": 0,
        "epoch": 1,
        "loss": math.pi * 3
    }
    for i in range(1, 101):
        log_out["progress"] = i
        log_out["epoch"] = int(log_out["progress"] / 25)
        log_out["loss"] = math.pi * 3 / i
        print(json.dumps(log_out))


if __name__ == "__main__":
    main()