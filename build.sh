#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

docker build -t $1 "$SCRIPTPATH"
