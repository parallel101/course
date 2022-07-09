ffmpeg -y -i "$1" -r 60 -filter_complex "[0:v]setpts=0.666667*PTS[v];[0:a]atempo=1.5[a]" -map "[v]" -map "[a]" /tmp/985211.mkv
