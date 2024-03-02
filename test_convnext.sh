CUDA_VISIBLE_DEVICES=0 python main.py --mode train --loc-eval --dataset mvtec --class-names all --multi-class --input-size 384 --condition-blocks 2 2 0 --semantic-cond --ratio-dynamic 4 \
        --quantize-enable --k-dynamic 32 --k-cond 512 --quantize-weight 1 --concat-dynamic --concat-dynamic --extractor convnext_xlarge_384_in22ft1k >> ./logs/test_convnext.log
        

        