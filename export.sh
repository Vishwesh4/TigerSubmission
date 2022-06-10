#!/usr/bin/env bash
DOCKERNAME="til_final"
./build.sh $DOCKERNAME

docker save $DOCKERNAME | xz -c > ../$DOCKERNAME.tar.xz
