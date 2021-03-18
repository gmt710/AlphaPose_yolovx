#!/bin/bash
nvidia-docker run -p 10001:22 -m 8GB -it --net=host \
		-v /home/gmt/pytorch:/root \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-e DISPLAY=unix$DISPLAY \
		-e GDK_SCALE \
		-e GDK_DPI_SCALE \
		--name cuda101_yolov5_v1_vis cuda101_yolov5:v1 /bin/bash
