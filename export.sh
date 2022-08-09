#!/usr/bin/env bash
DOCKERNAME="tiger_sri_biopsies"
./build.sh $DOCKERNAME

docker save $DOCKERNAME | xz -c > ../$DOCKERNAME.tar.xz
