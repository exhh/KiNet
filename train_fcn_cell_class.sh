gpu=1
data=PNET
model=ki67net

crop=200
datadir=datasets
batch=4
iterations=100000
lr=1e-3
momentum=0.99
num_cls=3
snapshot=10000
result=learned_models

outdir=${result}/${data}-${model}/${model}
mkdir -p ${result}/${data}-${model}

python train_fcn_cell_class.py ${outdir} --model ${model} \
    --num_cls ${num_cls} --gpu ${gpu} \
    --lr ${lr} -b ${batch} -m ${momentum} \
    --crop_size ${crop} --iterations ${iterations} \
    --augmentation \
    --snapshot ${snapshot} \
    --datadir ${datadir} \
    --dataset ${data} \
    --use_validation
