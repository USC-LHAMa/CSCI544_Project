# CSCI544_Project


Some link to helpful websites:
The Illustrated Transformer: http://jalammar.github.io/illustrated-transformer/

## Installation on Laptop/VM
- Download the code locally: `git clone https://github.com/USC-LHAMa/CSCI544_Project`
- Navigate to the code directory: `cd CSCI544_Project`
- Create Python virtual environment: `python -m venv .`
- Activate virtual environment: `source ./bin/activate`
- Install PyTorch: `pip install torch`
- Install tensorboardX: `pip install tensorboardX`
- Install other requirements: `pip install -r requirements.txt`
- Even more requirements!: `pip install -r requirements-dev.txt`
- Install LHAMa Transformers package: `python -m pip install -e .`
- Install tmux utility: `sudo apt-get install tmux`

## Run SQuAD Task
The command below runs SQuAD training and evaluation using the custom CNN model. `--model_type` can also be `lhamalstm`.

`python -m torch.distributed.launch --nproc_per_node=1 ./examples/run_squad.py --model_type lhamacnn --model_name_or_path bert-base-uncased --do_train --do_lower_case --do_eval --train_file input/train-v2.0.json --predict_file input/dev-v2.0.json --learning_rate 3e-5 --num_train_epochs 2 --max_seq_length 384 --doc_stride 128 --output_dir ../models/lhamacnn/ --per_gpu_eval_batch_size=3 --per_gpu_train_batch_size=3 --version_2_with_negative --local_rank=-1 --overwrite_output_dir --save_steps 5000`

For these types of long-running tasks, it is best to use tmux on the VM.
- Open a terminal on your laptop, e.g. default MacOS terminal or iTerm2
- SSH into the VM with your account: `gcloud compute ssh <user_name>@<vm_name>`
- Start tmux on the VM: `tmux`
- Navigate to the `transformers` directory and run the command to run the squad task
- If you are disconnected, SSH back into the server and attach to the previous session: `tmux attach-session -t <session_id>`
  - Existing tmux sessions can be viewed with `tmux ls`
  - If you connect to the wrong session: `tmux detach-client -s <attached_session_name>`
