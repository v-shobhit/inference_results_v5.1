import code.common.constants as C
import code.llmlib.fields as llm_fields
import code.fields.models as model_fields
import code.fields.loadgen as loadgen_fields
import code.fields.harness as harness_fields
import code.fields.triton as triton_fields
import os

os.environ['TLLM_NUMA_AWARE_WORKER_AFFINITY'] = '1'
os.environ['CUDA_SCALE_LAUNCH_QUEUES'] = '4x'
os.environ['TRTLLM_GEMM_ALLREDUCE_FUSION_ENABLED']= '1'
os.environ['TRTLLM_LLAMA_EAGER_FUSION_DISABLED']= '1'
os.environ['TRTLLM_DISABLE_NVFP4_LAYERNORM_FUSION']= '0'

EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {
        model_fields.gpu_batch_size: {
            'llama3.1-405b': 64,
        },
        loadgen_fields.min_query_count: 8313 * 10,
        model_fields.input_dtype: 'int32',
        llm_fields.llm_gen_config_path: 'code/llama3_1-405b/tensorrt/generation_config.json',
        loadgen_fields.min_duration: 600000,
        loadgen_fields.offline_expected_qps: 1.17 * 18,
        model_fields.precision: 'fp4',
        harness_fields.tensor_path: 'build/preprocessed_data/llama3.1-405b/',
        llm_fields.tensor_parallelism: 1,
        llm_fields.pipeline_parallelism: 4,
        llm_fields.trtllm_build_flags: {
            'max_beam_width': 1,
            'kv_cache_type': 'paged',
            'remove_input_padding': 'enable',
            'multiple_profiles': 'enable',
            'use_fused_mlp': 'enable',
            'context_fmha': 'enable',
            'max_num_tokens': 512,
            'max_input_len': 20000,
            'max_seq_len': 22000,
            'use_paged_context_fmha': 'enable',
            'tokens_per_block': 32,
            'use_fp8_context_fmha': 'enable',
            'norm_quant_fusion': 'enable',
            'gemm_allreduce_plugin': 'float16',
            'torch_compile_config': {
                'enable_fullgraph': True,
                'enable_piecewise_cuda_graph': True,
                'enable_userbuffers': False,
            }
        },
        llm_fields.trtllm_checkpoint_flags: {
            'kv_cache_dtype': 'fp8',
        },
        llm_fields.trtllm_runtime_flags: {
            'exclude_input_from_output': True,
            'use_inflight_batching': True,
            'max_num_tokens': 512,
            'num_postprocess_workers': 2,
            'batch_scheduler_policy': 'max_util',
            'context_chunking_policy': 'first_come_first_served',
            'kvcache_free_gpu_mem_frac': 0.95,
            'enable_chunked_context': True,
            'cuda_graph_batch_sizes': [1, 2, 4, 8, 16,32, 64, 96, 128], #play with cuda graph
            'cuda_graph_padding_enabled': True,
            'sampler_type': 'TRTLLMSampler',
            'num_postprocess_workers': 2,
        },
        harness_fields.use_graphs: False,
        llm_fields.use_token_latencies: True,
    },
    C.WorkloadSetting(C.HarnessType.Triton, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {
        model_fields.gpu_batch_size: {
            'llama3.1-405b': 128,
        },
        model_fields.input_dtype: 'int32',
        llm_fields.llm_gen_config_path: 'code/llama3_1-405b/tensorrt/generation_config.json',
        loadgen_fields.min_duration: 5400000,
        loadgen_fields.offline_expected_qps: 1.17,
        model_fields.precision: 'fp4',
        harness_fields.tensor_path: 'build/preprocessed_data/llama3.1-405b/',
        llm_fields.tensor_parallelism: 2,
        llm_fields.pipeline_parallelism: 2,
        llm_fields.trtllm_build_flags: {
            'max_beam_width': 1,
            'kv_cache_type': 'paged',
            'remove_input_padding': 'enable',
            'multiple_profiles': 'enable',
            'use_fused_mlp': 'enable',
            'context_fmha': 'enable',
            'max_num_tokens': 1536,
            'max_input_len': 20000,
            'max_seq_len': 22000,
            'use_paged_context_fmha': 'enable',
            'tokens_per_block': 32,
            'use_fp8_context_fmha': 'enable',
            'norm_quant_fusion': 'enable',
            'gemm_plugin': 'fp4',
        },
        llm_fields.trtllm_checkpoint_flags: {
            'kv_cache_dtype': 'fp8',
        },
        llm_fields.trtllm_runtime_flags: {
            'exclude_input_from_output': True,
            'use_inflight_batching': True,
            'max_num_tokens': 1536,
            'batch_scheduler_policy': 'max_util',
            'context_chunking_policy': 'first_come_first_served',
            'kvcache_free_gpu_mem_frac': 0.95,
            'enable_chunked_context': True,
        },
        harness_fields.use_graphs: False,
        llm_fields.use_token_latencies: True,
        triton_fields.use_triton: True,
    },
}
