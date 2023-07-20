#!/usr/bin/env python3
"""
4. How many by rocket?
"""
from datetime import datetime
import requests


if __name__ == '__main__':
    url = 'https://api.spacexdata.com/v4/launches'
    response = requests.get(url).json()
    rocketDict = {}
    for launch in response:
        rocket = launch.get('rocket')
        url = 'https://api.spacexdata.com/v4/rockets/{}'.format(rocket)
        response = requests.get(url).json()
        rocket = response.get('name')
        if rocketDict.get(rocket) is None:
            rocketDict[rocket] = 1
        else:
            rocketDict[rocket] += 1
    rocketList = sorted(rocketDict.items(), key=lambda kv: kv[0])
    rocketList = sorted(rocketList, key=lambda kv: kv[1], reverse=True)
    for rocket in rocketList:
        print("{}: {}".format(rocket[0], rocket[1]))
