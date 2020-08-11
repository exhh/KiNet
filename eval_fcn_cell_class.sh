gpu=3
testdata=PNET
traindata=PNET
model=ki67net
iteration=best
result=learned_models

datadir=datasets
num_cls=3
eval_result_folder=experiments

load_model=${result}/${traindata}-${model}/${model}-${iteration}.pth

python eval_fcn_cell_class.py ${load_model} --model ${model} \
    --num_cls ${num_cls} --gpu ${gpu} \
    --datadir ${datadir} \
    --dataset ${testdata} \
    --eval_result_folder ${eval_result_folder}
