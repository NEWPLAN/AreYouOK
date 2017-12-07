all:clean 
	cp -rf /mnt/hgfs/share/src ./
	chmod a-x ./src/*
	make -C src

run:all
	rm -rf core
	ulimit -c unlimited
	clear
	BCUBE_RANK=8 ./src/proc

share:
	make clean -C src
	rm -rf /mnt/hgfs/share/src
	cp -rf src /mnt/hgfs/share

clean:
	rm -rf src

