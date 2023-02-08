IMAGE_NAME=isles22_submission_pobotri
TAG=Task500
EXT=.tar.xz

time docker save $IMAGE_NAME:$TAG | xz -T0 -c > containers/${TAG}${EXT}