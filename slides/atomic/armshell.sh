#!/bin/bash

PORT=${1-2222}
ADDR=$(ip route get 1 | awk '{print $7;exit}')

echo -e "\033[32m-- Execute this command on your mobile:\033[0m\033[33;1m\n"
echo "p=\$(mktemp);socat pty,link=\$p,raw TCP:$ADDR:$PORT&(sleep .2&&(setsid \${SHELL-sh})<\$p>\$p 2>&1)"
echo -e "\n\033[0m\033[32m-- Listening at $ADDR:$PORT\033[0m\n"
stty -icanon -echo
nc -l -p $PORT
stty icanon echo
