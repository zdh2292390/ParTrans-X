# !/bin/bash
nepoch=100
for datapath in WN18 FB15K; do
	for k in 20 50 100 ;do
		for margin in 1 2 3 4;do
			for rate in 0.1 0.2 0.3 0.4; do
				for coreNum in 20; do
					resultpath="./"$datapath"/AdagradTransE_"$coreNum"thread_k"$k"_margin"$margin"_epoch"$nepoch"_rate"$rate

					if [ -d $resultpath ]; then
						rm -r $resultpath
					fi
					if [ ! -d $resultpath ]; then
						echo "mkdir "$resultpath
						mkdir $resultpath
						./ParAdagradTransE unif $coreNum $datapath $k $margin $nepoch $resultpath $rate adagrad> $resultpath/train.log
						if [ $? -eq 0 ];then
					    ./ParTest_TransE unif $coreNum $datapath $k $margin $nepoch $resultpath $rate adagrad > $resultpath/test.log
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