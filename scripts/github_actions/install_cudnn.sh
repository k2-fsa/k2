#!/bin/bash
#
# Copyright (c)  2020  Mobvoi Inc. (authors: Fangjun Kuang)

case $cuda in
  10.0)
    filename=cudnn-10.0-linux-x64-v7.6.5.32.tgz
    url="1H1MavvYdQosuhoTtjMpFOWUO1MiBMvC8"
    backup_url="https://bit.ly/3pDf2Nk"
    ;;
  10.1)
    filename=cudnn-10.1-linux-x64-v8.0.2.39.tgz
    url="13_IVjoRJmamBVOgEGSp1Tlr5nClZUqwj"
    backup_url="https://bit.ly/2IyDePU"
    ;;
  10.2)
    filename=cudnn-10.2-linux-x64-v8.0.2.39.tgz
    url="1beVinj771IPuqUsQovgTh5ZMqujBzb64"
    backup_url="https://bit.ly/2IDDX2p"
    ;;
  *)
    echo "Unsupported cuda version: $cuda"
    exit 1
    ;;
esac

gdown -q --id "$url" --output $filename
if [ ! -e $filename ]; then
  echo "Failed to download $filename"
  echo "Try $backup_url"
  curl -SL -o $filename "$backup_url"
fi

sudo tar xf ./$filename -C /usr/local
