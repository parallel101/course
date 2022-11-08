#!/bin/bash
set -e
rm -rf /tmp/a
rm -rf /tmp/b
mkdir -p /tmp/a/$(dirname $1)
mkdir -p /tmp/b/$(dirname $1)
cp $1 /tmp/a/$1
cp $1 /tmp/b/$1
cd /tmp
vim b/$1
diff -u a/$1 b/$1
