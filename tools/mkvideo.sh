#!/bin/bash

ffmpeg -r 30 -i %4d.png -vcodec libx264 out.avi
