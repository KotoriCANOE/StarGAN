export CUDA_VISIBLE_DEVICES=1
postfix=10
python graph.py --postfix $postfix --num-domains 6
python freeze_graph.py --input_binary False --input_graph model$postfix.tmp/model.graphdef --input_checkpoint model$postfix.tmp/model --output_graph model$postfix.tmp/model.pb --output_node_names Output

exit
