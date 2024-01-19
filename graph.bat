cd /d "%~dp0"

FOR %%i IN (14) DO (
	python graph.py --postfix %%i --num-domains 6
	python freeze_graph.py --input_binary False --input_graph model%%i.tmp\model.graphdef --input_checkpoint model%%i.tmp\model --output_graph model%%i.tmp\model.pb --output_node_names Output
)

pause
