#!/usr/bin/env python3
"""
2. Rate me is you can!
"""
import requests
import sys
import time


if __name__ == '__main__':
    url = sys.argv[1]
    response = requests.get(url)

    if response.status_code == 404:
        print("Not found")
    elif response.status_code == 403:
        rate_limit = int(response.headers['X-Ratelimit-Reset'])
        now = int(time.time())
        minutes = int((rate_limit - now) / 60)
        print("Reset in {} min".format(minutes))
    else:
        print(response.json()["location"])
