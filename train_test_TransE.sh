# !/bin/bash
nepoch=1000
for datapath in  WN18 FB15K; do
	for k in 20 50 100;do
		for margin in 1 2 3;do
			for rate in 0.01 0.001 0.1; do
				for coreNum in 20; do
					resultpath="./"$datapath"/TransE_"$coreNum"thread_k"$k"_margin"$margin"_epoch"$nepoch"_rate"$rate

					if [ -d $resultpath ]; then
						rm -r $resultpath
					fi

					if [ ! -d $resultpath ]; then
						echo "mkdir "$resultpath
						mkdir $resultpath
						./ParTransE unif $coreNum $datapath $k $margin $nepoch $resultpath $rate TransE> $resultpath/train.log
						if [ $? -eq 0 ];then
						    ./ParTest_TransE unif $coreNum $datapath $k $margin $nepoch $resultpath $rate TransE > $resultpath/test.log
						else 
						    exit
						fi

						if [ $? -eq 0 ];then
							continue
						else 
						    exit
						fi
					fi
				done
			done
		done		
	done
done



