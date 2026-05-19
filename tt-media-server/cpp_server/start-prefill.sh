export LLM_DEVICE_BACKEND=prefill 
export TT_IPC_SHM_C2P=tt_ipc_c2p
export TT_IPC_SHM_P2C=tt_ipc_p2c
./build/tt_media_server_cpp 

python src/runtime/runners/mock_prefill_runner.py 