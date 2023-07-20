#!/usr/bin/env python3
"""
3. What will be next?
"""
import requests


if __name__ == '__main__':
    url = "https://api.spacexdata.com/v4/launches/upcoming"
    response = requests.get(url)
    json = response.json()

    dates = [x['date_unix'] for x in json]
    index = dates.index(min(dates))
    next_launch = json[index]

    name = next_launch['name']
    date = next_launch['date_local']
    rocket = next_launch['rocket']
    launchpad = next_launch['launchpad']

    url_rocket = "https://api.spacexdata.com/v4/rockets/" + rocket
    req_rocket = requests.get(url_rocket)
    json_rocket = req_rocket.json()
    rocket_name = json_rocket['name']

    url_launchpad = "https://api.spacexdata.com/v4/launchpads/" + launchpad
    req_launchpad = requests.get(url_launchpad)
    json_launchpad = req_launchpad.json()
    launchpad_name = json_launchpad['name']
    launchpad_loc = json_launchpad['locality']

    info = (name + ' (' + date + ') ' + rocket_name + ' - ' +
            launchpad_name + ' (' + launchpad_loc + ')')
    print(info)
