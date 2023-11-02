## Use hdparm to test disk bandwidth

### Clear system cache
```shell
sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches   
```
### Test Read Bandwidth

Use **hdoarm** tool.
1. `sudo apt install hdparm`: Install hdparm tools
1. `lsblk`: List all the devices
1. `sudo hdparm -t /dev/sda`: Test Read Bandwidth

### Test Write Bandwidth

#### Use **fio** tool

Check [this](https://arstechnica.com/gadgets/2020/02/how-fast-are-your-disks-find-out-the-open-source-way-with-fio/) for details
1. **Random write** test:
``` shell
fio --name=random-write --ioengine=posixaio --rw=randwrite --bs=4k --numjobs=1 --size=4g --iodepth=1 --runtime=60 --time_based --end_fsync=1
```
2. **Sequential write** test:
``` shell
fio --name=random-write --ioengine=posixaio --rw=write --bs=4k --numjobs=1 --size=4g --iodepth=1 --runtime=60 --time_based --end_fsync=1
```

#### Use **dd** tool
Check [this](https://www.ibm.com/docs/en/spectrum-protect/8.1.9?topic=systems-analyzing-data-flow-dd-command)

1. Write test
``` shell
sudo time dd if=/dev/zero of=./draft bs=262144 count=40960  
```
2. Read test
``` shell
sudo time dd if=./draft of=/dev/null bs=262144 count=40960  
```