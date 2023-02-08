IMAGE_NAME=isles22_submission_pobotri
TAG_NAME=Task777
EXT=.tar.xz

docker build -t $IMAGE_NAME:$TAG_NAME . && time docker save $IMAGE_NAME:$TAG_NAME | xz -T0 -c > Task777_ISLES22_prep_DWI\(Multitask_sumloss\)-FINETUNE-BraTS2021_FLAIR_Necrotic\(Multitask_sumloss\)${EXT}