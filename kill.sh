OLD_RAY_PIDS=(
178452
415044
847782
860248
1388880
1449962
1779216
1898966
2008860
2176962
2283139
2342071
2689639
2946234
3082597
3111439
3197398
3233313
3303532
3335006
3364683
3420431
3676293
3755054
3777978
3787040
3807595
)

echo "===== 准备清理历史孤儿 Ray TaskRunner ====="

for pid in "${OLD_RAY_PIDS[@]}"; do
    if ! ps -p "$pid" >/dev/null 2>&1; then
        echo "SKIP $pid: 已不存在"
        continue
    fi

    ppid=$(ps -p "$pid" -o ppid= | tr -d ' ')
    cmd=$(ps -p "$pid" -o cmd=)

    if [[ "$ppid" == "1" && "$cmd" == *"ray::TaskRunner"* ]]; then
        echo "TERM $pid  PPID=$ppid  CMD=$cmd"
        kill -TERM "$pid"
    else
        echo "SKIP $pid: 当前状态已变化  PPID=$ppid  CMD=$cmd"
    fi
done