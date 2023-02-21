IMAGE_NAME=isles22_submission_pobotri
TAG="UNION_Task511_ISLES_d_m_div2_dwionly_ENSEMBLE_Task762_Task763_Task764_Task765"

time docker build -t $IMAGE_NAME:$TAG . 

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
VOLUME_SUFFIX=$(dd if=/dev/urandom bs=32 count=1 | md5sum | cut --delimiter=' ' --fields=1)
MEM_LIMIT="15g"  # Maximum is currently 30g, configurable in your algorithm image settings on grand challenge
docker volume create $IMAGE_NAME-output-$VOLUME_SUFFIX

# Do not change any of the parameters to docker run, these are fixed
docker run --rm \
        --memory="${MEM_LIMIT}" \
        --memory-swap="${MEM_LIMIT}" \
        --network="none" \
        --cap-drop="ALL" \
        --security-opt="no-new-privileges" \
        --shm-size="128m" \
        --pids-limit="256" \
        --gpus 1 \
        -v $SCRIPTPATH/test/:/input/ \
        -v $IMAGE_NAME-output-$VOLUME_SUFFIX:/output/ \
        $IMAGE_NAME:$TAG

docker run --rm \
        -v $IMAGE_NAME-output-$VOLUME_SUFFIX:/output/ \
        -v $SCRIPTPATH/test/:/input/ \
        python:3.9-slim python -c "import json, sys; f1 = json.load(open('/output/results.json')); f2 = json.load(open('/input/expected_output.json')); sys.exit(f1 != f2);"

if [ $? -eq 0 ]; then
    echo "Tests successfully passed..."
else
    echo "Expected output was not found..."
    exit
fi
docker volume rm $IMAGE_NAME-output-$VOLUME_SUFFIX
echo "Exporting ${IMAGE_NAME}:${TAG}"
time docker save $IMAGE_NAME:$TAG | xz -T0 -c > containers/${TAG}.tar.xz
