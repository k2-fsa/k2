#!/usr/bin/env python3
# Copyright    2022  Xiaomi Corp.        (authors: Fangjun Kuang)

import subprocess
from datetime import datetime, timedelta


def get_last_commit_date() -> datetime:
    date = (
        subprocess.check_output(["git", "log", "-1", "--format=%ad", "--date=unix"])
        .decode("ascii")
        .strip()
    )
    return datetime.utcfromtimestamp(int(date))


def main():
    last_commit_date_utc = get_last_commit_date()
    now_utc = datetime.utcnow()
    if last_commit_date_utc + timedelta(days=1) > now_utc:
        print("true")
    else:
        print("false")


if __name__ == "__main__":
    main()
