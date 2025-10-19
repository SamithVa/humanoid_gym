sudo docker run -it \
-v ./:/data \
--device /dev/dri \
--group-add video \
--volume /tmp/.X11-unix:/tmp/.X11-unix \
--env="DISPLAY=$DISPLAY" \
--env="QT_X11_NO_MITSHM=1" \
--name=pinu20 \
ubuntu20_pino_cro:v0 /bin/bash