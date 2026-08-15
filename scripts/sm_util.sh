while true; do
  dcgmi dmon -e 1002 -i 1 -c 1 | awk '$1=="GPU" && $2=="1" {print $3}'
  sleep 1
done | ttyplot -s 1 -t "SM_ACTIVE gpu1"
