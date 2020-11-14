#!/usr/bin/env python3
#
# Copyright (c)  2020  Mobvoi Inc. (authors: Fangjun Kuang)

import os
from mediafire.client import MediaFireClient


def main():
    email = os.environ.get('username')
    password = os.environ.get('password')

    if email is None or password is None:
        msg = 'Please set username and password on GitHub by following '
        msg += 'https://docs.github.com/en/free-pro-team@latest/actions/reference/authentication-in-a-workflow'  # noqa
        raise Exception(msg)

    cuda_version = os.environ['cuda']
    client = MediaFireClient()
    client.login(email=email, password=password, app_id='42511')
    files = {
        '10.0': 'cudnn-10.0-linux-x64-v7.6.5.32.tgz',
        '10.1': 'cudnn-10.1-linux-x64-v8.0.2.39.tgz',
        '10.2': 'cudnn-10.2-linux-x64-v8.0.2.39.tgz'
    }

    filename = files[cuda_version]
    print('Downloading {}'.format(filename))
    client.download_file("mf:/cudnn/{}".format(filename), './')

    os.system('ls -lh ./')

    print('Installing {}'.format(filename))
    os.system('sudo tar xf ./{} -C /usr/local'.format(filename))
    print('Done!')


if __name__ == '__main__':
    main()
