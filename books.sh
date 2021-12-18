echo '课件：https://github.com/parallel101/course'
echo '作业：https://github.com/parallel101/hw02'
grep -- '- \[' README.md | sed 's/- \[//g' | sed 's/](/：/g' | sed 's/)$//g'
