#!/bin/bash

set -e
i=1000
for x in $*; do
    i=$[$i + 1]
    ffmpeg -i $x -vcodec copy -acodec copy -vbsf h264_mp4toannexb /tmp/t_$i.ts
done
cat /tmp/t_*.ts > /tmp/a_a.ts
ffmpeg -y -i /tmp/a_a.ts -acodec copy -vcodec copy -absf aac_adtstoasc /tmp/out.mp4
rm -f /tmp/a_a.ts /tmp/t_*.ts
