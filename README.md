The source code for the algorithm container for threshold_model, generated with
evalutils version 0.3.1.

This repository contains a Docker generation example, in order to submit an algorithm to ISLES22 challenge.


Docker submission tutorial to isles22 (set quality to HD):
https://www.youtube.com/watch?v=lLPS_XnzmgM


* Submission code debugging
    
    `docker compose build && docker compose run -d debug`

* Final test

    `docker build -t isles22_submission_pobotri . && bash test.sh`

* Build and compress to *.xz file

    `docker build -t isles22_submission_pobotri . && time docker save isles22_submission_pobotri | xz -T0 -c > Task770_ISLES_preprocessed_DWI.tar.xz`
