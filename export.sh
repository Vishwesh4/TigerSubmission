#!/usr/bin/env bash
DOCKERNAME="tiger_sri_resection_chunk"
# DOCKERNAME="tiger_sri_biopsy_chunk"
./build.sh $DOCKERNAME

docker save $DOCKERNAME | xz -c > ../$DOCKERNAME.tar.xz
