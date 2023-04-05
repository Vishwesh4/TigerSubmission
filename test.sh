#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
DOCKERNAME="tiger_sri_resection_chunk"
# DOCKERNAME="tiger_sri_biopsy_chunk"

SEGMENTATION_FILE="/output/images/breast-cancer-segmentation-for-tils/segmentation.tif"
DETECTION_FILE="/output/detected-lymphocytes.json"
TILS_SCORE_FILE="/output/til-score.json"

MEMORY=30g

echo "Building docker"
./build.sh $DOCKERNAME

echo "Creating volume..."
docker volume create tiger-output

echo "Running algorithm..."
docker run --rm \
        --memory=$MEMORY \
        --gpus=all\
        --memory-swap=$MEMORY \
        --network=none \
        --cap-drop=ALL \
        --security-opt="no-new-privileges" \
        --shm-size=128m \
        --pids-limit=256 \
        -v /home/vishwesh/Projects/testinputs/testinput_TC_B103/:/input/ \
        -v tiger-output:/output/ \
        $DOCKERNAME

# -v /home/vishwesh/Projects/testinputs/testinput_TC_B103/:/input/ \
# -v /home/vishwesh/Projects/testinputs/testinput_104S/:/input/ \
## -v /home/vishwesh/Projects/testinputs/testinput_52/:/input/ \
echo "Checking output files..."
docker run --rm \
        -v tiger-output:/output/ \
        python:3.8-slim \
        python -m json.tool $TILS_SCORE_FILE; \
        /bin/bash; \
        [[ -f $SEGMENTATION_FILE ]] || printf 'Expected file %s does not exist!\n' "$SEGMENTATION_FILE"; \
        [[ -f $TILS_SCORE_FILE ]] || printf 'Expected file %s does not exist!\n' "$TILS_SCORE_FILE"; \


echo "Removing volume..."
docker volume rm tiger-output
